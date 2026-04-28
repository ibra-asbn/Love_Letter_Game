"""
Behavioral cloning for an actor that consumes belief probabilities directly.

This is the first "true" actor-belief communication experiment:
obs -> encoder -> hidden
hidden -> belief probabilities
concat(hidden, belief probabilities) -> actor -> action

The actor loss flows through the belief probabilities, so the belief module is
not only an auxiliary teacher; it becomes part of the decision path.
"""

from pathlib import Path
import argparse
import json
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefHead,
    LATENT,
    NUM_CARDS,
    NUM_OPPONENTS,
)
from love_letter.paths import checkpoint_path, data_path
from scripts.evaluation.evaluate_models import evaluate_config
from love_letter.belief_actor import BeliefConditionedPolicy


def load_dataset(max_samples=None):
    with open(data_path("heuristic_dataset.pkl"), "rb") as f:
        data = pickle.load(f)

    obs = data["obs"]
    mask = data["mask"]
    action = data["action"]
    hidden_cards = data["hidden_cards"]

    if max_samples is not None and max_samples < len(obs):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(obs), size=max_samples, replace=False)
        obs = obs[idx]
        mask = mask[idx]
        action = action[idx]
        hidden_cards = hidden_cards[idx]

    return (
        torch.as_tensor(obs, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        torch.as_tensor(action, dtype=torch.long),
        torch.as_tensor(hidden_cards, dtype=torch.long),
    )


def split_dataset(obs, mask, action, hidden_cards, val_ratio=0.1):
    n = len(obs)
    n_val = max(1, int(n * val_ratio))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = TensorDataset(obs[train_idx], mask[train_idx], action[train_idx], hidden_cards[train_idx])
    val_ds = TensorDataset(obs[val_idx], mask[val_idx], action[val_idx], hidden_cards[val_idx])
    return train_ds, val_ds


def evaluate_validation(encoder, belief_head, actor, val_loader, device):
    encoder.eval()
    belief_head.eval()
    actor.eval()

    action_correct = 0
    action_total = 0
    belief_correct = 0
    belief_total = 0
    action_loss_sum = 0.0
    belief_loss_sum = 0.0
    ce_action = nn.CrossEntropyLoss(reduction="sum")
    ce_belief = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    with torch.no_grad():
        for obs_b, mask_b, act_b, hidden_b in val_loader:
            obs_b = obs_b.to(device)
            mask_b = mask_b.to(device)
            act_b = act_b.to(device)
            hidden_b = hidden_b.to(device)
            h = encoder.forward_hidden(obs_b)
            belief_logits, belief_probs = belief_head(h)
            logits = actor(h, belief_probs, mask_b)

            action_loss_sum += float(ce_action(logits, act_b).item())
            belief_loss_sum += float(
                ce_belief(
                    belief_logits.reshape(-1, NUM_CARDS),
                    hidden_b.reshape(-1),
                ).item()
            )

            action_correct += int((logits.argmax(dim=-1) == act_b).sum().item())
            action_total += len(act_b)

            preds = belief_logits.argmax(dim=-1)
            valid = hidden_b != -1
            belief_correct += int((preds[valid] == hidden_b[valid]).sum().item())
            belief_total += int(valid.sum().item())

    encoder.train()
    belief_head.train()
    actor.train()
    return {
        "val_action_loss": action_loss_sum / max(1, action_total),
        "val_belief_loss": belief_loss_sum / max(1, belief_total),
        "val_action_acc": action_correct / max(1, action_total),
        "val_belief_acc": belief_correct / max(1, belief_total),
    }


def save_checkpoint(path, encoder, belief_head, actor, metrics):
    torch.save(
        {
            "model_type": "belief_conditioned_actor_v1",
            "encoder": encoder.state_dict(),
            "belief_head": belief_head.state_dict(),
            "actor": actor.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def run_arena(policy, n_games=100):
    configs = {
        "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
        "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
        "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
        "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
    }
    return {
        name: {k: float(v) for k, v in evaluate_config(policy, opponents, n_games=n_games).items()}
        for name, opponents in configs.items()
    }


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading heuristic dataset...")
    obs, mask, action, hidden_cards = load_dataset(args.max_samples)
    print(f"  samples: {len(obs):,}")
    train_ds, val_ds = split_dataset(obs, mask, action, hidden_cards)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    encoder = BeliefConditionedEncoder().to(device)
    belief_head = BeliefHead().to(device)
    actor = BeliefConditionedActor().to(device)

    params = list(encoder.parameters()) + list(belief_head.parameters()) + list(actor.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ce_action = nn.CrossEntropyLoss()
    ce_belief = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = -1.0
    output_path = checkpoint_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        belief_head.train()
        actor.train()
        total_action_loss = 0.0
        total_belief_loss = 0.0
        total = 0
        correct = 0

        for obs_b, mask_b, act_b, hidden_b in train_loader:
            obs_b = obs_b.to(device)
            mask_b = mask_b.to(device)
            act_b = act_b.to(device)
            hidden_b = hidden_b.to(device)

            h = encoder.forward_hidden(obs_b)
            belief_logits, belief_probs = belief_head(h)
            logits = actor(h, belief_probs, mask_b)

            action_loss = ce_action(logits, act_b)
            belief_loss = ce_belief(
                belief_logits.reshape(-1, NUM_CARDS),
                hidden_b.reshape(-1),
            )
            loss = action_loss + args.belief_coef * belief_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optim.step()

            batch_size = len(act_b)
            total += batch_size
            total_action_loss += float(action_loss.item()) * batch_size
            total_belief_loss += float(belief_loss.item()) * batch_size
            correct += int((logits.argmax(dim=-1) == act_b).sum().item())

        val = evaluate_validation(encoder, belief_head, actor, val_loader, device)
        metrics = {
            "epoch": epoch,
            "train_action_loss": total_action_loss / max(1, total),
            "train_belief_loss": total_belief_loss / max(1, total),
            "train_action_acc": correct / max(1, total),
            **val,
        }
        history.append(metrics)
        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={metrics['train_action_acc']:.3f} "
            f"val_acc={metrics['val_action_acc']:.3f} "
            f"belief_acc={metrics['val_belief_acc']:.3f} "
            f"loss={metrics['val_action_loss']:.3f}"
        )

        if metrics["val_action_acc"] > best_val_acc:
            best_val_acc = metrics["val_action_acc"]
            save_checkpoint(output_path, encoder, belief_head, actor, {"history": history, "best": metrics})
            print(f"  saved best -> {output_path}")

    policy = BeliefConditionedPolicy(encoder.cpu(), belief_head.cpu(), actor.cpu()).eval()
    arena = run_arena(policy, n_games=args.eval_games)
    report_path = Path("logs/evaluations") / f"{output_path.stem}_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(output_path),
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "history": history,
        "arena": arena,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Eval report -> {report_path}")
    print(json.dumps(arena, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--belief-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--output", default="belief_conditioned_bc.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
