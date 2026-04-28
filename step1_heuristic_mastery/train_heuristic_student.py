"""Train a recurrent student to imitate HeuristicBot without overfitting."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefHead,
    LATENT,
    NUM_CARDS,
)
from love_letter.paths import checkpoint_path
from step1_heuristic_mastery.common import (
    ExperimentLogger,
    STEP_CHECKPOINT_DIR,
    STEP_DATA_DIR,
    STEP_REPORT_DIR,
    ensure_step_dirs,
    now_stamp,
    resolve_step_path,
)


def resolve_checkpoint(name_or_path):
    if not name_or_path:
        return None
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = checkpoint_path(name_or_path)
    if candidate.exists():
        return candidate
    candidate = STEP_CHECKPOINT_DIR / name_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")


def load_or_create_model(start=None):
    encoder = BeliefConditionedEncoder()
    belief_head = BeliefHead()
    actor = BeliefConditionedActor()
    ckpt = None
    if start:
        start_path = resolve_checkpoint(start)
        ckpt = torch.load(start_path, map_location="cpu", weights_only=True)
        if ckpt.get("model_type") != "belief_conditioned_actor_v1":
            raise ValueError(f"{start_path} is not a belief-conditioned actor checkpoint")
        encoder.load_state_dict(ckpt["encoder"])
        belief_head.load_state_dict(ckpt["belief_head"])
        actor.load_state_dict(ckpt["actor"])
    return ckpt, encoder, belief_head, actor


def load_dataset(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def split_by_game(sequences, val_ratio, test_ratio, seed):
    games = sorted({int(seq["game"]) for seq in sequences})
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n = len(games)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_games = set(games[:n_test])
    val_games = set(games[n_test : n_test + n_val])
    train_games = set(games[n_test + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for seq in sequences:
        game = int(seq["game"])
        if game in test_games:
            splits["test"].append(seq)
        elif game in val_games:
            splits["val"].append(seq)
        elif game in train_games:
            splits["train"].append(seq)
    return splits, {
        "train_games": len(train_games),
        "val_games": len(val_games),
        "test_games": len(test_games),
        "train_sequences": len(splits["train"]),
        "val_sequences": len(splits["val"]),
        "test_sequences": len(splits["test"]),
    }


def collate_sequences(batch):
    max_len = max(len(seq["steps"]) for seq in batch)
    batch_size = len(batch)
    obs = np.zeros((batch_size, max_len, 158), dtype=np.float32)
    masks = np.zeros((batch_size, max_len, 1000), dtype=np.bool_)
    actions = np.zeros((batch_size, max_len), dtype=np.int64)
    hidden_cards = np.full((batch_size, max_len, 3), -1, dtype=np.int64)
    valid = np.zeros((batch_size, max_len), dtype=np.bool_)
    teacher_seats = np.zeros(batch_size, dtype=np.int64)

    for i, seq in enumerate(batch):
        teacher_seats[i] = int(seq.get("teacher_seats_in_game", -1))
        for t, step in enumerate(seq["steps"]):
            obs[i, t] = step["obs"]
            masks[i, t] = step["mask"].astype(bool)
            actions[i, t] = step["action"]
            hidden_cards[i, t] = step["hidden_cards"]
            valid[i, t] = True

    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "masks": torch.as_tensor(masks, dtype=torch.bool),
        "actions": torch.as_tensor(actions, dtype=torch.long),
        "hidden_cards": torch.as_tensor(hidden_cards, dtype=torch.long),
        "valid": torch.as_tensor(valid, dtype=torch.bool),
        "teacher_seats": torch.as_tensor(teacher_seats, dtype=torch.long),
    }


def forward_sequences(encoder, belief_head, actor, batch, device):
    obs = batch["obs"].to(device)
    masks = batch["masks"].to(device)
    valid = batch["valid"].to(device)
    batch_size, max_len, _ = obs.shape
    hidden = torch.zeros(batch_size, LATENT, dtype=obs.dtype, device=device)
    action_logits = []
    belief_logits = []

    for t in range(max_len):
        hidden_next = encoder.forward_hidden(obs[:, t], hidden)
        active = valid[:, t].unsqueeze(-1)
        hidden = torch.where(active, hidden_next, hidden)
        b_logits, b_probs = belief_head(hidden)
        logits = actor(hidden, b_probs, masks[:, t])
        action_logits.append(logits)
        belief_logits.append(b_logits)

    return torch.stack(action_logits, dim=1), torch.stack(belief_logits, dim=1)


def sequence_metrics(action_logits, belief_logits, batch, ce_action=None, ce_belief=None):
    valid = batch["valid"]
    actions = batch["actions"]
    hidden_cards = batch["hidden_cards"]
    flat_valid = valid.reshape(-1)
    flat_logits = action_logits.reshape(-1, action_logits.shape[-1])
    flat_actions = actions.reshape(-1)
    preds = flat_logits.argmax(dim=-1)
    correct = (preds[flat_valid] == flat_actions[flat_valid]).float()

    metrics = {
        "action_acc": float(correct.mean().item()) if correct.numel() else 0.0,
        "samples": int(flat_valid.sum().item()),
    }
    if ce_action is not None:
        metrics["action_loss"] = float(ce_action(flat_logits[flat_valid], flat_actions[flat_valid]).item())

    flat_belief_logits = belief_logits.reshape(-1, 3, NUM_CARDS)
    flat_hidden_cards = hidden_cards.reshape(-1, 3)
    belief_valid = flat_hidden_cards.reshape(-1) != -1
    if belief_valid.any():
        belief_preds = flat_belief_logits.argmax(dim=-1)
        belief_correct = (belief_preds.reshape(-1)[belief_valid] == flat_hidden_cards.reshape(-1)[belief_valid]).float()
        metrics["belief_acc"] = float(belief_correct.mean().item())
        if ce_belief is not None:
            metrics["belief_loss"] = float(
                ce_belief(flat_belief_logits.reshape(-1, NUM_CARDS), flat_hidden_cards.reshape(-1)).item()
            )
    else:
        metrics["belief_acc"] = 0.0
        metrics["belief_loss"] = 0.0
    return metrics


def evaluate_split(encoder, belief_head, actor, loader, device):
    encoder.eval()
    belief_head.eval()
    actor.eval()
    ce_action = nn.CrossEntropyLoss()
    ce_belief = nn.CrossEntropyLoss(ignore_index=-1)
    totals = Counter()
    weighted = Counter()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            action_logits, belief_logits = forward_sequences(encoder, belief_head, actor, batch, device)
            metrics = sequence_metrics(action_logits, belief_logits, batch, ce_action, ce_belief)
            n = metrics["samples"]
            totals["samples"] += n
            for key, value in metrics.items():
                if key != "samples":
                    weighted[key] += value * n

    return {key: float(value / max(1, totals["samples"])) for key, value in weighted.items()} | {
        "samples": int(totals["samples"])
    }


def save_checkpoint(path, encoder, belief_head, actor, metadata, base_ckpt=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "belief_conditioned_actor_v1",
        "encoder": encoder.cpu().state_dict(),
        "belief_head": belief_head.cpu().state_dict(),
        "actor": actor.cpu().state_dict(),
        "metadata": metadata,
    }
    if base_ckpt and "critic" in base_ckpt:
        payload["critic"] = base_ckpt["critic"]
    torch.save(payload, path)


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Train the step1 heuristic student.")
    parser.add_argument("--dataset", default="teacher_sequences_attempt1.pkl")
    parser.add_argument("--start", default=None, help="Optional belief-conditioned checkpoint to continue from.")
    parser.add_argument("--output", default="heuristic_student_attempt1.pth")
    parser.add_argument("--report", default="heuristic_student_attempt1_train.json")
    parser.add_argument("--run-log", default="step1_heuristic_mastery/logs/2026-04-24_step1_train_attempt1.md")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--belief-coef", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--test-ratio", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    logger = ExperimentLogger(args.run_log)
    logger.reset()

    dataset_path = resolve_step_path(args.dataset, STEP_DATA_DIR)
    dataset = load_dataset(dataset_path)
    splits, split_summary = split_by_game(dataset["sequences"], args.val_ratio, args.test_ratio, args.seed)

    base_ckpt, encoder, belief_head, actor = load_or_create_model(args.start)
    encoder.to(device)
    belief_head.to(device)
    actor.to(device)
    initial_actor = deepcopy(actor).to(device).eval()

    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(belief_head.parameters()) + list(actor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    ce_action = nn.CrossEntropyLoss()
    ce_belief = nn.CrossEntropyLoss(ignore_index=-1)
    output = resolve_step_path(args.output, STEP_CHECKPOINT_DIR)
    report = resolve_step_path(args.report, STEP_REPORT_DIR)

    logger.write(
        "Debut entrainement imitation heuristique",
        expected=(
            "Apprendre HeuristicBot en recurrent, en surveillant validation/test par parties "
            "pour eviter l'overfitting."
        ),
        actual=f"dataset={dataset_path}, start={args.start}, output={output}",
        details={"args": vars(args), "dataset_summary": dataset["summary"], "split_summary": split_summary},
    )

    initial_val = evaluate_split(encoder, belief_head, initial_actor, val_loader, device)
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        belief_head.train()
        actor.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            action_logits, belief_logits = forward_sequences(encoder, belief_head, actor, batch, device)
            flat_valid = batch["valid"].reshape(-1)
            action_loss = ce_action(
                action_logits.reshape(-1, action_logits.shape[-1])[flat_valid],
                batch["actions"].reshape(-1)[flat_valid],
            )
            belief_loss = ce_belief(
                belief_logits.reshape(-1, NUM_CARDS),
                batch["hidden_cards"].reshape(-1),
            )
            loss = action_loss + args.belief_coef * belief_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(belief_head.parameters()) + list(actor.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics = evaluate_split(encoder, belief_head, actor, train_loader, device)
        val_metrics = evaluate_split(encoder, belief_head, actor, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": train_metrics,
            "val": val_metrics,
            "overfit_gap_action_acc": train_metrics["action_acc"] - val_metrics["action_acc"],
        }
        history.append(row)
        improved = val_metrics["action_acc"] > best_val_acc + args.min_delta
        if improved:
            best_val_acc = val_metrics["action_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "encoder": deepcopy(encoder.cpu().state_dict()),
                "belief_head": deepcopy(belief_head.cpu().state_dict()),
                "actor": deepcopy(actor.cpu().state_dict()),
            }
            encoder.to(device)
            belief_head.to(device)
            actor.to(device)
            save_checkpoint(
                output,
                encoder,
                belief_head,
                actor,
                metadata={
                    "created_at": now_stamp(),
                    "step": "heuristic_mastery",
                    "best_epoch": best_epoch,
                    "best_val_action_acc": best_val_acc,
                    "args": vars(args),
                    "dataset_summary": dataset["summary"],
                    "split_summary": split_summary,
                    "history": history,
                },
                base_ckpt=base_ckpt,
            )
        else:
            epochs_without_improvement += 1

        logger.write(
            f"Epoch {epoch}/{args.epochs}",
            expected="La validation doit monter puis se stabiliser; l'ecart train-val doit rester raisonnable.",
            actual=(
                f"val_acc={val_metrics['action_acc']:.4f}, "
                f"train_acc={train_metrics['action_acc']:.4f}, "
                f"gap={row['overfit_gap_action_acc']:.4f}, improved={improved}"
            ),
            details=row,
        )

        if epochs_without_improvement >= args.patience:
            logger.write(
                "Early stopping",
                expected="Arreter quand la validation ne progresse plus pour eviter l'overfitting.",
                actual=f"best_epoch={best_epoch}, best_val_action_acc={best_val_acc:.4f}",
            )
            break

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        belief_head.load_state_dict(best_state["belief_head"])
        actor.load_state_dict(best_state["actor"])
        encoder.to(device)
        belief_head.to(device)
        actor.to(device)

    final_train = evaluate_split(encoder, belief_head, actor, train_loader, device)
    final_val = evaluate_split(encoder, belief_head, actor, val_loader, device)
    final_test = evaluate_split(encoder, belief_head, actor, test_loader, device)
    summary = {
        "created_at": now_stamp(),
        "dataset": str(dataset_path),
        "output": str(output),
        "start": args.start,
        "initial_val": initial_val,
        "best_epoch": best_epoch,
        "best_val_action_acc": best_val_acc,
        "final_train": final_train,
        "final_val": final_val,
        "final_test": final_test,
        "overfit_gap_train_val": final_train["action_acc"] - final_val["action_acc"],
        "history": history,
        "dataset_summary": dataset["summary"],
        "split_summary": split_summary,
        "args": vars(args),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin entrainement imitation heuristique",
        expected="Sauvegarder le meilleur checkpoint validation, pas le dernier.",
        actual=(
            f"best_epoch={best_epoch}, val_acc={final_val['action_acc']:.4f}, "
            f"test_acc={final_test['action_acc']:.4f}, checkpoint={output}"
        ),
        details=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

