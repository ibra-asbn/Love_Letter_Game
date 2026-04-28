"""Train the Step5 fast Chancellor execution head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_rl_finetune.common import ExperimentLogger, now_stamp
from step5_execution_heads.chancellor_head import (
    ChancellorExecutionHead,
    chancellor_choice_features,
    decode_chancellor_choice,
)


STEP5_DIR = PROJECT_ROOT / "step5_execution_heads"
DATASET_DIR = STEP5_DIR / "datasets"
CHECKPOINT_DIR = STEP5_DIR / "checkpoints"
REPORT_DIR = STEP5_DIR / "reports"
LOG_DIR = STEP5_DIR / "logs"


def ensure_dirs() -> None:
    for path in [DATASET_DIR, CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_dataset(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path, DATASET_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset not found: {path}")


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    return [record for record in payload if record.get("kind") == "chancellor_choice"]


def row_score(row: dict) -> float:
    return float(row.get("score", row.get("winrate", 0.0)))


def build_rows(records: list[dict], args) -> tuple[dict, dict]:
    observations = []
    features = []
    targets = []
    weights = []
    groups = []
    actions = []
    is_model = []
    decoded = []
    group_summaries = []

    for group_id, record in enumerate(records):
        if "observation" not in record:
            raise ValueError("Dataset must include `observation`; rerun collect_execution_teacher after Step5 patch.")
        candidates = record.get("candidate_rows", [])
        if len(candidates) <= 1:
            continue
        model_action = int(record["model_action"])
        model_rows = [row for row in candidates if int(row["action"]) == model_action]
        if not model_rows:
            continue
        model_score = row_score(model_rows[0])
        target_values = []
        action_values = []
        for row in candidates:
            action = int(row["action"])
            target = row_score(row) - model_score
            if action == model_action:
                target = 0.0
            action_values.append(action)
            target_values.append(float(target))

        best_idx = int(np.argmax(target_values))
        best_action = int(action_values[best_idx])
        best_adv = float(target_values[best_idx])
        if args.only_clear_regret and not bool(record.get("clear_regret")):
            continue

        for action, target in zip(action_values, target_values):
            abs_target = abs(float(target))
            if int(action) == model_action:
                weight = args.model_weight
            elif abs_target < args.tie_threshold:
                weight = args.tie_weight
            else:
                weight = min(args.max_weight, 1.0 + abs_target / max(1e-6, args.tie_threshold))
            observations.append(np.asarray(record["observation"], dtype=np.float32))
            features.append(chancellor_choice_features(record["pool"], int(action), int(record["deck_remaining"])))
            targets.append(float(target))
            weights.append(float(weight))
            groups.append(int(group_id))
            actions.append(int(action))
            is_model.append(bool(int(action) == model_action))
            decoded.append(decode_chancellor_choice(record["pool"], int(action)))

        group_summaries.append(
            {
                "group": int(group_id),
                "model_action": model_action,
                "best_action": best_action,
                "best_advantage": best_adv,
                "clear_regret": bool(record.get("clear_regret")),
                "phase": record.get("phase"),
                "pool": record.get("pool"),
                "candidate_count": len(candidates),
            }
        )

    if not observations:
        raise RuntimeError("No trainable Chancellor rows found.")

    arrays = {
        "observations": np.stack(observations).astype(np.float32),
        "features": np.stack(features).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "groups": np.asarray(groups, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "is_model": np.asarray(is_model, dtype=bool),
    }
    summary = {
        "states": int(len(group_summaries)),
        "rows": int(len(targets)),
        "clear_regret_states": int(sum(1 for row in group_summaries if row["clear_regret"])),
        "positive_override_states": int(
            sum(1 for row in group_summaries if row["best_action"] != row["model_action"] and row["best_advantage"] >= args.eval_margin)
        ),
        "mean_best_advantage": float(np.mean([row["best_advantage"] for row in group_summaries])),
        "sample_groups": group_summaries[:20],
        "sample_rows": [
            {
                "group": int(groups[i]),
                "action": int(actions[i]),
                "decoded": decoded[i],
                "target": float(targets[i]),
                "weight": float(weights[i]),
            }
            for i in range(min(30, len(targets)))
        ],
    }
    return arrays, summary


def split_groups(groups: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_groups = set(int(x) for x in unique[:n_val])
    val_mask = np.asarray([int(group) in val_groups for group in groups], dtype=bool)
    train_mask = ~val_mask
    return np.where(train_mask)[0], np.where(val_mask)[0]


def group_metrics(head: ChancellorExecutionHead, data: dict, row_idx: np.ndarray, margin: float, device) -> dict:
    head.eval()
    obs = torch.as_tensor(data["observations"][row_idx], dtype=torch.float32, device=device)
    feat = torch.as_tensor(data["features"][row_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = head(obs, feat).detach().cpu().numpy()
    targets = data["targets"][row_idx]
    groups = data["groups"][row_idx]
    actions = data["actions"][row_idx]
    mae = float(np.mean(np.abs(pred - targets))) if len(row_idx) else 0.0
    top1 = []
    override_agreement = []
    pred_overrides = []
    target_overrides = []
    for group in np.unique(groups):
        positions = np.where(groups == group)[0]
        p = pred[positions]
        t = targets[positions]
        a = actions[positions]
        model_positions = np.where(np.abs(t) < 1e-8)[0]
        exact_model_positions = np.where(data["is_model"][row_idx][positions])[0]
        if len(exact_model_positions):
            model_positions = exact_model_positions
        model_pos = int(model_positions[0]) if len(model_positions) else 0
        pred_pos = int(np.argmax(p))
        target_pos = int(np.argmax(t))
        pred_override = pred_pos != model_pos and (p[pred_pos] - p[model_pos]) >= margin
        target_override = target_pos != model_pos and (t[target_pos] - t[model_pos]) >= margin
        top1.append(int(a[pred_pos] == a[target_pos]))
        override_agreement.append(int(pred_override == target_override))
        pred_overrides.append(int(pred_override))
        target_overrides.append(int(target_override))
    return {
        "rows": int(len(row_idx)),
        "groups": int(len(np.unique(groups))) if len(row_idx) else 0,
        "mae": mae,
        "top1_acc": float(np.mean(top1)) if top1 else 0.0,
        "override_agreement": float(np.mean(override_agreement)) if override_agreement else 0.0,
        "pred_override_rate": float(np.mean(pred_overrides)) if pred_overrides else 0.0,
        "target_override_rate": float(np.mean(target_overrides)) if target_overrides else 0.0,
    }


def train(data: dict, args, logger: ExperimentLogger):
    device = torch.device(args.device)
    train_idx, val_idx = split_groups(data["groups"], args.val_ratio, args.seed + 71)
    head = ChancellorExecutionHead(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_t = TensorDataset(torch.as_tensor(train_idx, dtype=torch.long))
    loader = DataLoader(train_t, batch_size=args.batch_size, shuffle=True)
    history = []
    logger.write(
        "Debut entrainement tete Chancelier",
        expected="Apprendre les avantages relatifs au choix actuel sans rollouts a l'inference.",
        actual=f"train_rows={len(train_idx)}, val_rows={len(val_idx)}",
        details={
            "initial_train": group_metrics(head, data, train_idx, args.eval_margin, device),
            "initial_val": group_metrics(head, data, val_idx, args.eval_margin, device),
        },
    )
    for epoch in range(1, args.epochs + 1):
        losses = []
        for (idx,) in loader:
            idx = idx.to(device)
            obs = torch.as_tensor(data["observations"][idx.cpu().numpy()], dtype=torch.float32, device=device)
            feat = torch.as_tensor(data["features"][idx.cpu().numpy()], dtype=torch.float32, device=device)
            target = torch.as_tensor(data["targets"][idx.cpu().numpy()], dtype=torch.float32, device=device)
            weight = torch.as_tensor(data["weights"][idx.cpu().numpy()], dtype=torch.float32, device=device)
            pred = head(obs, feat)
            huber = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
            loss = (huber * weight).sum() / weight.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": group_metrics(head, data, train_idx, args.eval_margin, device),
            "val": group_metrics(head, data, val_idx, args.eval_margin, device),
        }
        history.append(row)
        logger.write(
            f"Epoch tete Chancelier {epoch}/{args.epochs}",
            expected="Val top1/override agreement montent sans taux d'override delirant.",
            actual=(
                f"loss={row['loss']:.4f}, val_top1={row['val']['top1_acc']:.3f}, "
                f"val_override={row['val']['pred_override_rate']:.3f}, "
                f"val_mae={row['val']['mae']:.3f}"
            ),
            details=row,
        )
    return head, history, train_idx, val_idx


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train Step5 Chancellor execution head.")
    parser.add_argument("--dataset", default="chancellor_teacher_attempt1.json")
    parser.add_argument("--output", default="chancellor_head_attempt1.pth")
    parser.add_argument("--report", default="chancellor_head_attempt1_train.json")
    parser.add_argument("--run-log", default="step5_execution_heads/logs/2026-04-26_chancellor_head_attempt1_train.md")
    parser.add_argument("--only-clear-regret", action="store_true")
    parser.add_argument("--tie-threshold", type=float, default=0.035)
    parser.add_argument("--tie-weight", type=float, default=0.20)
    parser.add_argument("--model-weight", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=4.0)
    parser.add_argument("--eval-margin", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.06)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=71000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    dataset_path = resolve_dataset(args.dataset)
    records = load_records(dataset_path)
    data, dataset_summary = build_rows(records, args)
    logger.write(
        "Dataset tete Chancelier charge",
        expected="Utiliser uniquement les etats Chancelier naturels labellises par oracle CRN.",
        actual=f"states={dataset_summary['states']}, rows={dataset_summary['rows']}",
        details={"dataset": str(dataset_path), "summary": dataset_summary, "args": vars(args)},
    )
    head, history, train_idx, val_idx = train(data, args, logger)

    output_path = Path(args.output)
    if output_path.parent == Path("."):
        output_path = CHECKPOINT_DIR / output_path
    payload = {
        "model_type": "step5_chancellor_execution_head",
        "created_at": now_stamp(),
        "head": head.cpu().state_dict(),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "metadata": {
            "args": vars(args),
            "dataset": str(dataset_path),
            "dataset_summary": dataset_summary,
            "history": history,
            "final_train": group_metrics(head, data, train_idx, args.eval_margin, torch.device("cpu")),
            "final_val": group_metrics(head, data, val_idx, args.eval_margin, torch.device("cpu")),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    report = {
        "created_at": now_stamp(),
        "dataset": str(dataset_path),
        "checkpoint": str(output_path),
        "dataset_summary": dataset_summary,
        "history": history,
        "final_train": payload["metadata"]["final_train"],
        "final_val": payload["metadata"]["final_val"],
    }
    report_path = Path(args.report)
    if report_path.parent == Path("."):
        report_path = REPORT_DIR / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin entrainement tete Chancelier",
        expected="Sauvegarder une tete rapide candidate a evaluer en arena.",
        actual=f"checkpoint={output_path}, report={report_path}",
        details={"final_val": report["final_val"]},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
