"""Train a fast regret-override head on top of the step2 policy.

The rollout-guided teacher is useful but too slow for play: it searches at
decision time. This script turns that teacher into a small override head that
keeps the base actor by default and only changes action when a learned gate is
confident.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from love_letter.belief_actor import ACTION_DIM, BELIEF_DIM, LATENT
from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS, random_action
from step3_action_value.evaluate_rollout_guided import evaluate_candidate_actions
from step3_action_value.mini_rollout_probe import choose_actions_for_probe, classify_state, decode_action


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


class RegretOverrideHead(nn.Module):
    """Small policy patch: predict an override action plus a keep/override gate."""

    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        input_dim = LATENT + BELIEF_DIM + ACTION_DIM
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, belief, model_action, action_mask=None):
        action_one_hot = torch.zeros(
            hidden.shape[0],
            ACTION_DIM,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        action_one_hot.scatter_(1, model_action.view(-1, 1), 1.0)
        x = torch.cat([hidden, belief.reshape(hidden.shape[0], -1), action_one_hot], dim=-1)
        features = self.trunk(x)
        logits = self.action_head(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        gate_logit = self.gate_head(features).squeeze(-1)
        return logits, gate_logit


class Player0FeaturePolicy:
    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None

    def act(self, obs_dict):
        action, self.state = self.policy.act(obs_dict, self.state, agent_id="player_0")
        hidden = self.state.detach().cpu().squeeze(0).numpy()
        debug = getattr(self.policy, "last_debug", None)
        belief = _debug_belief_array(debug)
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        return int(action), hidden.astype(np.float32), belief.astype(np.float32)


def _debug_belief_array(debug):
    if debug is None:
        return None
    belief = getattr(debug, "belief_probs", None)
    if belief is None:
        belief = getattr(debug, "probs", None)
    if belief is None:
        return None
    if isinstance(belief, torch.Tensor):
        return belief.detach().cpu().squeeze(0).numpy()
    return np.asarray(belief, dtype=np.float32)


def opponent_action(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def collect_override_labels(args, checkpoint, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    category_counts = Counter()
    config_counts = Counter()
    label_counts = Counter()
    regret_by_category = defaultdict(list)
    config_names = args.collect_configs
    target_per_category = {category: args.states_per_category for category in args.categories}

    for game in range(args.collect_games):
        if all(category_counts[category] >= target_per_category[category] for category in args.categories):
            break

        config_name = config_names[game % len(config_names)]
        opponents = OPPONENT_CONFIGS[config_name]
        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        policy = Player0FeaturePolicy(checkpoint)

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                category = classify_state(env, agent)
                model_action, hidden, belief = policy.act(obs_dict)

                if (
                    category in args.categories
                    and category_counts[category] < target_per_category[category]
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    heuristic_action = int(bot.choose_action(env, agent))
                    candidates = choose_actions_for_probe(env, args.max_actions)
                    for forced in [model_action, heuristic_action]:
                        if forced not in candidates:
                            candidates = [forced] + candidates
                    candidates = list(dict.fromkeys(candidates))[: args.max_actions]

                    eval_args = argparse.Namespace(
                        rollouts_per_action=args.rollouts_per_action,
                        player0_continuation=args.player0_continuation,
                    )
                    rows = evaluate_candidate_actions(
                        env,
                        candidates,
                        checkpoint,
                        opponents,
                        eval_args,
                        decision_seed=seed * 100 + turn,
                    )
                    by_action = {row["action"]: row for row in rows}
                    model_row = by_action.get(model_action)
                    best = rows[0]
                    regret = float(best["winrate"] - model_row["winrate"]) if model_row else 0.0
                    reward_regret = (
                        float(best["mean_reward"] - model_row["mean_reward"]) if model_row else 0.0
                    )
                    changed = int(
                        best["action"] != model_action
                        and regret >= args.override_margin
                        and reward_regret >= args.min_reward_margin
                    )
                    target_action = int(best["action"] if changed else model_action)

                    category_counts[category] += 1
                    config_counts[config_name] += 1
                    label_counts["changed" if changed else "keep"] += 1
                    regret_by_category[category].append(regret)

                    records.append(
                        {
                            "category": category,
                            "config": config_name,
                            "seed": seed,
                            "turn": turn,
                            "hidden": hidden,
                            "belief": belief,
                            "mask": obs_dict["action_mask"].astype(np.bool_),
                            "model_action": int(model_action),
                            "target_action": target_action,
                            "changed": changed,
                            "best_action": int(best["action"]),
                            "model_winrate": float(model_row["winrate"]) if model_row else None,
                            "best_winrate": float(best["winrate"]),
                            "regret_winrate": regret,
                            "reward_regret": reward_regret,
                            "top_actions": rows[: min(5, len(rows))],
                        }
                    )

                    if len(records) % args.log_every_states == 0:
                        logger.write(
                            "Collecte override regret",
                            expected="Construire des labels larges mais seulement avec marge robuste.",
                            actual=(
                                f"states={len(records)}, changed={label_counts['changed']}, "
                                f"categories={dict(category_counts)}"
                            ),
                            details={
                                "config_counts": dict(config_counts),
                                "label_counts": dict(label_counts),
                            },
                        )

                env.step(model_action)
            else:
                env.step(opponent_action(env, agent, obs_dict, opponents, bot))

    if not records:
        raise RuntimeError("No override labels collected")

    changed_records = [record for record in records if record["changed"]]
    summary = {
        "states": len(records),
        "category_counts": dict(category_counts),
        "config_counts": dict(config_counts),
        "label_counts": dict(label_counts),
        "change_rate": float(label_counts["changed"] / len(records)),
        "mean_regret": float(np.mean([record["regret_winrate"] for record in records])),
        "mean_changed_regret": float(np.mean([record["regret_winrate"] for record in changed_records]))
        if changed_records
        else 0.0,
        "mean_regret_by_category": {
            category: float(np.mean(values)) for category, values in regret_by_category.items()
        },
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


def evaluate_head(head, data, gate_threshold=0.5):
    head.eval()
    with torch.no_grad():
        logits, gate_logit = head(data["hidden"], data["belief"], data["model_action"], data["mask"])
        pred_action = logits.argmax(dim=-1)
        gate_prob = torch.sigmoid(gate_logit)
        pred_changed = (gate_prob >= gate_threshold) & (pred_action != data["model_action"])
        changed = data["changed"].bool()
        final_action = torch.where(pred_changed, pred_action, data["model_action"])
        metrics = {
            "target_acc": float((final_action == data["target_action"]).float().mean().item()),
            "action_target_acc": float((pred_action == data["target_action"]).float().mean().item()),
            "gate_acc": float((pred_changed == changed).float().mean().item()),
            "true_change_rate": float(changed.float().mean().item()),
            "pred_change_rate": float(pred_changed.float().mean().item()),
            "changed_target_acc": float((final_action[changed] == data["target_action"][changed]).float().mean().item())
            if changed.any()
            else None,
            "keep_acc": float((final_action[~changed] == data["model_action"][~changed]).float().mean().item())
            if (~changed).any()
            else None,
            "mean_gate_prob_changed": float(gate_prob[changed].mean().item()) if changed.any() else None,
            "mean_gate_prob_keep": float(gate_prob[~changed].mean().item()) if (~changed).any() else None,
        }
    head.train()
    return metrics


def train_head(records, args, logger):
    device = torch.device(args.device)
    data = tensors_from_records(records, device)
    head = RegretOverrideHead(hidden_dim=args.hidden_dim).to(device)

    n = len(records)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 77)
    perm = torch.randperm(n, generator=generator, device=device)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    action_loss_fn = nn.CrossEntropyLoss(reduction="none")
    gate_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    history = []

    logger.write(
        "Debut entrainement tete override",
        expected="Apprendre une correction prudente sans remplacer Step2.",
        actual=f"train={len(train_idx)}, val={len(val_idx)}",
        details={"initial_all": evaluate_head(head, data, gate_threshold=args.gate_threshold)},
    )

    for epoch in range(1, args.epochs + 1):
        losses = []
        for (idx,) in loader:
            logits, gate_logit = head(
                data["hidden"][idx],
                data["belief"][idx],
                data["model_action"][idx],
                data["mask"][idx],
            )
            changed = data["changed"][idx]
            regret = data["regret"][idx]
            action_weights = 1.0 + changed * (args.changed_action_weight - 1.0)
            action_weights = action_weights * (1.0 + args.regret_weight * regret)
            action_loss = (action_loss_fn(logits, data["target_action"][idx]) * action_weights).mean()

            gate_weights = 1.0 + changed * (args.changed_gate_weight - 1.0)
            gate_loss = (gate_loss_fn(gate_logit, changed) * gate_weights).mean()
            loss = action_loss + args.gate_loss_weight * gate_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        train_data = {key: value[train_idx] for key, value in data.items()}
        val_data = {key: value[val_idx] for key, value in data.items()}
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": evaluate_head(head, train_data, gate_threshold=args.gate_threshold),
            "val": evaluate_head(head, val_data, gate_threshold=args.gate_threshold),
            "all": evaluate_head(head, data, gate_threshold=args.gate_threshold),
        }
        history.append(row)
        logger.write(
            f"Epoch override {epoch}/{args.epochs}",
            expected="Le gate doit rester selectif et toucher les vrais regrets.",
            actual=(
                f"loss={row['loss']:.4f}, val_target={row['val']['target_acc']:.3f}, "
                f"val_pred_change={row['val']['pred_change_rate']:.3f}"
            ),
            details=row,
        )

    return head, history


def jsonable_records(records, limit=60):
    out = []
    for record in records[:limit]:
        compact = {key: value for key, value in record.items() if key not in {"hidden", "belief", "mask"}}
        compact["model_decoded"] = decode_action(compact["model_action"])
        compact["target_decoded"] = decode_action(compact["target_action"])
        compact["best_decoded"] = decode_action(compact["best_action"])
        out.append(compact)
    return out


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train a regret-override head for step3.")
    parser.add_argument("--start", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--output", default="step3_regret_override_broad_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=3500)
    parser.add_argument("--states-per-category", type=int, default=55)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["guard", "priest", "spy", "king", "prince", "chancellor_card", "baron"],
    )
    parser.add_argument("--collect-configs", nargs="+", default=list(OPPONENT_CONFIGS.keys()))
    parser.add_argument("--rollouts-per-action", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--override-margin", type=float, default=0.12)
    parser.add_argument("--min-reward-margin", type=float, default=-999.0)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--changed-action-weight", type=float, default=6.0)
    parser.add_argument("--changed-gate-weight", type=float, default=4.0)
    parser.add_argument("--regret-weight", type=float, default=4.0)
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--log-every-states", type=int, default=30)
    parser.add_argument("--report", default="step3_regret_override_broad_attempt1_train.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_regret_override_broad_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9800)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.start)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut step3 regret override large",
        expected=(
            "Transformer le teacher rollout-guided large en correcteur rapide: "
            "garder Step2 sauf quand le regret est robuste."
        ),
        actual=f"start={checkpoint}",
        details=vars(args),
    )

    records, collection_summary = collect_override_labels(args, checkpoint, logger)
    logger.write(
        "Dataset override collecte",
        expected="Obtenir des corrections sur les cartes a fort manque a gagner.",
        actual=(
            f"states={collection_summary['states']}, "
            f"change_rate={collection_summary['change_rate']:.2%}, "
            f"mean_changed_regret={collection_summary['mean_changed_regret']:.3f}"
        ),
        details=collection_summary,
    )

    head, history = train_head(records, args, logger)

    output = Path(args.output)
    if output.parent == Path("."):
        output = CHECKPOINT_DIR / output
    payload = {
        "model_type": "step3_regret_override_v1",
        "created_at": now_stamp(),
        "base_checkpoint": str(checkpoint),
        "head": head.cpu().state_dict(),
        "head_hidden_dim": args.hidden_dim,
        "categories": args.categories,
        "metadata": {
            "args": vars(args),
            "collection_summary": collection_summary,
            "history": history,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

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
        "Fin step3 regret override large",
        expected="Sauvegarder une tete d'override candidate a evaluer en arena.",
        actual=f"checkpoint={output}, report={report_path}",
        details={
            "collection_summary": collection_summary,
            "final_metrics": history[-1] if history else None,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
