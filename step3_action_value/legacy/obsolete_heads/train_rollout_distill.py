"""Distill rollout action-values into the step2 actor.

This is the first real step3 attempt: sample critical states, evaluate candidate
actions by determinized rollouts, and train the actor to prefer high-margin
rollout winners while staying close to the step2 policy elsewhere.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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

from love_letter.belief_actor import BeliefConditionedActor, BeliefConditionedEncoder, BeliefHead, LATENT
from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step3_action_value.mini_rollout_probe import (
    CRITICAL_ORDER,
    choose_actions_for_probe,
    classify_state,
    decode_action,
    evaluate_actions,
    random_action,
)


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs():
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "belief_conditioned_actor_v1":
        raise ValueError(f"{path} is not a belief-conditioned checkpoint")
    encoder = BeliefConditionedEncoder()
    belief_head = BeliefHead()
    actor = BeliefConditionedActor()
    encoder.load_state_dict(ckpt["encoder"])
    belief_head.load_state_dict(ckpt["belief_head"])
    actor.load_state_dict(ckpt["actor"])
    return ckpt, encoder, belief_head, actor


class Player0CollectorPolicy:
    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None
        self.last_hidden = None
        self.last_belief = None

    def act(self, obs_dict):
        action, self.state = self.policy.act(obs_dict, self.state, agent_id="player_0")
        self.last_hidden = self.state.detach().cpu().squeeze(0).numpy()
        debug = getattr(self.policy, "last_debug", None)
        self.last_belief = debug.belief_probs.detach().cpu().squeeze(0).numpy()
        return int(action)


def opponent_action(env, agent, obs_dict, mode, bot):
    if mode == "heuristic":
        return bot.choose_action(env, agent)
    if mode == "random":
        return random_action(obs_dict)
    if mode == "mixed":
        return bot.choose_action(env, agent) if np.random.rand() < 0.5 else random_action(obs_dict)
    raise ValueError(mode)


def collect_rollout_labels(args, checkpoint, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    category_counts = Counter()
    label_counts = Counter()

    for game in range(args.collect_games):
        if all(category_counts[c] >= args.states_per_category for c in args.categories):
            break
        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        policy = Player0CollectorPolicy(checkpoint)

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                category = classify_state(env, agent)
                action = policy.act(obs_dict)

                if (
                    category in args.categories
                    and category_counts[category] < args.states_per_category
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    actions = choose_actions_for_probe(env, args.max_actions)
                    if action not in actions:
                        actions = [action] + actions[:-1]
                    rows = evaluate_actions(env, actions, args, checkpoint)
                    by_action = {row["action"]: row for row in rows}
                    model_row = by_action.get(action)
                    best = rows[0]
                    regret = best["winrate"] - model_row["winrate"] if model_row else 0.0
                    target = best["action"] if regret >= args.min_label_margin else action
                    changed = int(target != action)
                    label_counts["changed" if changed else "kept_model"] += 1
                    category_counts[category] += 1

                    records.append(
                        {
                            "category": category,
                            "seed": seed,
                            "hidden": policy.last_hidden,
                            "belief": policy.last_belief,
                            "mask": obs_dict["action_mask"].astype(np.bool_),
                            "model_action": int(action),
                            "target_action": int(target),
                            "changed": changed,
                            "best_action": int(best["action"]),
                            "best_winrate": float(best["winrate"]),
                            "model_winrate": float(model_row["winrate"]) if model_row else None,
                            "regret_winrate": float(regret),
                            "top_actions": rows[: min(5, len(rows))],
                        }
                    )

                    if len(records) % args.log_every_states == 0:
                        logger.write(
                            "Collecte labels rollout",
                            expected="Accumuler des etats critiques avec labels a marge claire.",
                            actual=(
                                f"states={len(records)}, changed={label_counts['changed']}, "
                                f"categories={dict(category_counts)}"
                            ),
                        )

                env.step(action)
            else:
                env.step(opponent_action(env, agent, obs_dict, args.collect_opponents, bot))

    if not records:
        raise RuntimeError("No rollout labels collected")

    summary = {
        "states": len(records),
        "category_counts": dict(category_counts),
        "label_counts": dict(label_counts),
        "change_rate": float(label_counts["changed"] / len(records)),
        "mean_regret": float(np.mean([r["regret_winrate"] for r in records])),
        "mean_changed_regret": float(np.mean([r["regret_winrate"] for r in records if r["changed"]]))
        if label_counts["changed"]
        else 0.0,
    }
    return records, summary


def tensors_from_records(records, device):
    return {
        "hidden": torch.as_tensor(np.array([r["hidden"] for r in records]), dtype=torch.float32, device=device),
        "belief": torch.as_tensor(np.array([r["belief"] for r in records]), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.array([r["mask"] for r in records]), dtype=torch.bool, device=device),
        "model_action": torch.as_tensor([r["model_action"] for r in records], dtype=torch.long, device=device),
        "target_action": torch.as_tensor([r["target_action"] for r in records], dtype=torch.long, device=device),
        "changed": torch.as_tensor([r["changed"] for r in records], dtype=torch.float32, device=device),
        "regret": torch.as_tensor([r["regret_winrate"] for r in records], dtype=torch.float32, device=device),
    }


def evaluate_actor(actor, data):
    actor.eval()
    with torch.no_grad():
        logits = actor(data["hidden"], data["belief"], data["mask"])
        pred = logits.argmax(dim=-1)
        changed = data["changed"].bool()
        metrics = {
            "target_acc": float((pred == data["target_action"]).float().mean().item()),
            "model_acc": float((pred == data["model_action"]).float().mean().item()),
            "changed_target_acc": float((pred[changed] == data["target_action"][changed]).float().mean().item())
            if changed.any()
            else None,
            "unchanged_model_acc": float((pred[~changed] == data["model_action"][~changed]).float().mean().item())
            if (~changed).any()
            else None,
            "pred_changed_rate": float((pred != data["model_action"]).float().mean().item()),
        }
    actor.train()
    return metrics


def train_actor(actor, records, args, device, logger):
    data = tensors_from_records(records, device)
    old_actor = deepcopy(actor).to(device).eval()
    for param in old_actor.parameters():
        param.requires_grad_(False)

    n = len(records)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 123)
    perm = torch.randperm(n, generator=rng, device=device)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    history = []

    logger.write(
        "Debut entrainement rollout distill",
        expected="Corriger les decisions a regret sans detruire le comportement step2.",
        actual=f"train={len(train_idx)}, val={len(val_idx)}",
        details={"initial_all": evaluate_actor(actor, data)},
    )

    for epoch in range(1, args.epochs + 1):
        losses = []
        for (idx,) in loader:
            logits = actor(data["hidden"][idx], data["belief"][idx], data["mask"][idx])
            weights = 1.0 + data["changed"][idx] * (args.changed_weight - 1.0)
            weights = weights * (1.0 + args.regret_weight * data["regret"][idx])
            loss = (ce(logits, data["target_action"][idx]) * weights).mean()

            if args.kl_coef > 0:
                with torch.no_grad():
                    old_logits = old_actor(data["hidden"][idx], data["belief"][idx], data["mask"][idx])
                    old_probs = torch.softmax(old_logits, dim=-1)
                    old_log_probs = torch.log_softmax(old_logits, dim=-1)
                new_log_probs = torch.log_softmax(logits, dim=-1)
                kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
                loss = loss + args.kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        val_data = {k: v[val_idx] for k, v in data.items()}
        train_data = {k: v[train_idx] for k, v in data.items()}
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": evaluate_actor(actor, train_data),
            "val": evaluate_actor(actor, val_data),
            "all": evaluate_actor(actor, data),
        }
        history.append(row)
        logger.write(
            f"Epoch {epoch}/{args.epochs}",
            expected="changed_target_acc doit monter avec pred_changed_rate controle.",
            actual=(
                f"loss={row['loss']:.4f}, val_target={row['val']['target_acc']:.3f}, "
                f"val_changed={row['val']['changed_target_acc']}"
            ),
            details=row,
        )
    return history


def save_checkpoint(path, base_ckpt, encoder, belief_head, actor, metadata):
    payload = {
        "model_type": "belief_conditioned_actor_v1",
        "encoder": encoder.cpu().state_dict(),
        "belief_head": belief_head.cpu().state_dict(),
        "actor": actor.cpu().state_dict(),
        "metadata": metadata,
    }
    if "critic" in base_ckpt:
        payload["critic"] = base_ckpt["critic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def jsonable_records(records, limit=50):
    out = []
    for record in records[:limit]:
        compact = {
            key: value
            for key, value in record.items()
            if key not in {"hidden", "belief", "mask"}
        }
        compact["model_decoded"] = decode_action(compact["model_action"])
        compact["target_decoded"] = decode_action(compact["target_action"])
        compact["best_decoded"] = decode_action(compact["best_action"])
        out.append(compact)
    return out


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train step3 actor from rollout action-values.")
    parser.add_argument("--start", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--output", default="step3_rollout_distilled_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=2500)
    parser.add_argument("--states-per-category", type=int, default=40)
    parser.add_argument("--categories", nargs="+", default=CRITICAL_ORDER)
    parser.add_argument("--rollouts-per-action", type=int, default=24)
    parser.add_argument("--max-actions", type=int, default=18)
    parser.add_argument("--min-label-margin", type=float, default=0.06)
    parser.add_argument("--collect-opponents", choices=["heuristic", "random", "mixed"], default="mixed")
    parser.add_argument("--opponent-policy", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--player0-continuation", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--changed-weight", type=float, default=4.0)
    parser.add_argument("--regret-weight", type=float, default=3.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--log-every-states", type=int, default=25)
    parser.add_argument("--report", default="step3_rollout_distilled_attempt1_train.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-24_step3_rollout_distill_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9300)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args.start)
    base_ckpt, encoder, belief_head, actor = load_checkpoint(checkpoint)
    encoder.to(device).eval()
    belief_head.to(device).eval()
    actor.to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    for param in belief_head.parameters():
        param.requires_grad_(False)

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut step3 rollout distillation",
        expected=(
            "Produire un checkpoint actor brut qui garde step2 mais corrige les decisions "
            "a regret d'apres rollouts."
        ),
        actual=f"start={checkpoint}",
        details=vars(args),
    )

    records, collection_summary = collect_rollout_labels(args, checkpoint, logger)
    logger.write(
        "Dataset rollout collecte",
        expected="Obtenir assez de labels changes pour apprendre un correctif utile.",
        actual=(
            f"states={collection_summary['states']}, "
            f"change_rate={collection_summary['change_rate']:.2%}, "
            f"mean_regret={collection_summary['mean_regret']:.4f}"
        ),
        details=collection_summary,
    )
    history = train_actor(actor, records, args, device, logger)

    output = Path(args.output)
    if output.parent == Path("."):
        output = CHECKPOINT_DIR / output
    metadata = {
        "source_checkpoint": str(checkpoint),
        "training": "step3_rollout_action_value_distillation",
        "args": vars(args),
        "collection_summary": collection_summary,
        "history": history,
    }
    save_checkpoint(output, base_ckpt, encoder, belief_head, actor, metadata)

    report = {
        "created_at": now_stamp(),
        "start": str(checkpoint),
        "output": str(output),
        "collection_summary": collection_summary,
        "history": history,
        "sample_records": jsonable_records(records),
    }
    report_path = Path(args.report)
    if report_path.parent == Path("."):
        report_path = REPORT_DIR / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin step3 rollout distillation",
        expected="Sauvegarder un checkpoint candidat a evaluer.",
        actual=f"checkpoint={output}, report={report_path}",
        details={
            "collection_summary": collection_summary,
            "final_metrics": history[-1] if history else None,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

