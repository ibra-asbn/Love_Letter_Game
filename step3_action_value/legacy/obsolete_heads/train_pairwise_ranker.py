"""Train a Step3 action-value ranker from rollout comparisons.

This is the more serious Step3 attempt: keep Step2 as the base policy, collect
candidate actions in tactical states, evaluate every candidate with
determinized rollouts, then train a small ranker to predict which candidate is
better in that exact state. The ranker learns values for actions, not hard
"replace with this one" labels.
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

from love_letter.belief_actor import BELIEF_DIM, LATENT, OBS_DIM
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS
from step3_action_value.evaluate_rollout_guided import evaluate_candidate_actions
from step3_action_value.mini_rollout_probe import choose_actions_for_probe, classify_state, decode_action
from step3_action_value.train_action_value_head import action_parts
from step3_action_value.train_regret_override import Player0FeaturePolicy, opponent_action


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def candidate_actions(env, model_action: int, heuristic_action: int, max_actions: int) -> list[int]:
    """Use one consistent candidate generator for training and inference."""
    candidates = choose_actions_for_probe(env, max_actions)
    for forced in [model_action, heuristic_action]:
        if forced not in candidates:
            candidates = [forced] + candidates
    return list(dict.fromkeys(int(a) for a in candidates))[:max_actions]


def rollout_confidence_weight(env, args) -> float:
    """Weight late/decisive states more because short-horizon rollout labels are cleaner."""
    deck_progress = 1.0 - (len(env._deck) / 21.0)
    active_players = sum(1 for agent in env.possible_agents if not env.terminations.get(agent, False))
    elimination_progress = (4 - active_players) / 3.0
    played = sum(len(cards) for cards in env._played_cards.values()) / 16.0
    weight = 1.0
    weight += args.stage_weight * max(0.0, min(1.0, deck_progress))
    weight += args.active_weight * max(0.0, min(1.0, elimination_progress))
    weight += args.played_weight * max(0.0, min(1.0, played))
    return float(weight)


class PairwiseActionRanker(nn.Module):
    def __init__(self, hidden_dim: int = 256, embed_dim: int = 24):
        super().__init__()
        self.card_emb = nn.Embedding(11, embed_dim)
        self.target_emb = nn.Embedding(10, embed_dim // 2)
        self.guess_emb = nn.Embedding(10, embed_dim // 2)
        action_dim = embed_dim + embed_dim // 2 + embed_dim // 2
        input_dim = OBS_DIM + LATENT + BELIEF_DIM + action_dim * 3 + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _embed_action(self, actions: torch.Tensor) -> torch.Tensor:
        card, target, guess = action_parts(actions)
        return torch.cat(
            [self.card_emb(card), self.target_emb(target), self.guess_emb(guess)],
            dim=-1,
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        belief: torch.Tensor,
        candidate_action: torch.Tensor,
        model_action: torch.Tensor,
        heuristic_action: torch.Tensor,
    ) -> torch.Tensor:
        candidate_emb = self._embed_action(candidate_action)
        model_emb = self._embed_action(model_action)
        heuristic_emb = self._embed_action(heuristic_action)
        flags = torch.stack(
            [
                (candidate_action == model_action).float(),
                (candidate_action == heuristic_action).float(),
                (candidate_action // 100 == model_action // 100).float(),
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                obs,
                hidden,
                belief.reshape(hidden.shape[0], -1),
                candidate_emb,
                model_emb,
                heuristic_emb,
                flags,
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def collect_ranker_states(args, checkpoint: Path, logger: ExperimentLogger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    category_counts = Counter()
    pair_counts = Counter()
    config_counts = Counter()
    regret_by_category = defaultdict(list)
    changed_by_category = defaultdict(int)
    stage_weights = []

    def target_reached() -> bool:
        if args.states_per_category_config:
            return all(
                pair_counts[(category, config)] >= args.states_per_category_config
                for category in args.categories
                for config in args.collect_configs
            )
        return all(category_counts[category] >= args.states_per_category for category in args.categories)

    def can_collect(category: str, config_name: str) -> bool:
        if args.states_per_category_config:
            return pair_counts[(category, config_name)] < args.states_per_category_config
        return category_counts[category] < args.states_per_category

    for game in range(args.collect_games):
        if target_reached():
            break

        config_name = args.collect_configs[game % len(args.collect_configs)]
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
                    and can_collect(category, config_name)
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    heuristic_action = int(bot.choose_action(env, agent))
                    candidates = candidate_actions(env, model_action, heuristic_action, args.max_actions)
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
                    if model_row is None:
                        env.step(model_action)
                        continue
                    best = rows[0]
                    regret = float(best["winrate"] - model_row["winrate"])
                    score_values = [
                        float(row["winrate"] + args.reward_score_weight * row["mean_reward"])
                        for row in rows
                    ]
                    record = {
                        "category": category,
                        "config": config_name,
                        "seed": seed,
                        "turn": turn,
                        "obs": obs_dict["observation"].astype(np.float32),
                        "hidden": hidden.astype(np.float32),
                        "belief": belief.astype(np.float32),
                        "model_action": int(model_action),
                        "heuristic_action": int(heuristic_action),
                        "actions": [int(row["action"]) for row in rows],
                        "targets": score_values,
                        "winrates": [float(row["winrate"]) for row in rows],
                        "mean_rewards": [float(row["mean_reward"]) for row in rows],
                        "best_action": int(best["action"]),
                        "model_winrate": float(model_row["winrate"]),
                        "best_winrate": float(best["winrate"]),
                        "regret_winrate": regret,
                        "deck_size": int(len(env._deck)),
                        "active_players": int(
                            sum(1 for player in env.possible_agents if not env.terminations.get(player, False))
                        ),
                        "played_cards_count": int(sum(len(cards) for cards in env._played_cards.values())),
                        "stage_weight": rollout_confidence_weight(env, args),
                        "top_actions": rows[: min(5, len(rows))],
                    }
                    records.append(record)
                    category_counts[category] += 1
                    pair_counts[(category, config_name)] += 1
                    config_counts[config_name] += 1
                    regret_by_category[category].append(regret)
                    stage_weights.append(record["stage_weight"])
                    if best["action"] != model_action and regret >= args.teacher_margin:
                        changed_by_category[category] += 1

                    if len(records) % args.log_every_states == 0:
                        logger.write(
                            "Collecte ranker Step3",
                            expected="Accumuler des etats tactiques values par rollouts avec plusieurs actions candidates.",
                            actual=f"states={len(records)}, categories={dict(category_counts)}",
                            details={
                                "config_counts": dict(config_counts),
                                "pair_counts": {
                                    f"{category}|{config}": count
                                    for (category, config), count in sorted(pair_counts.items())
                                },
                                "changed_by_category": dict(changed_by_category),
                            },
                        )
                env.step(model_action)
            else:
                env.step(opponent_action(env, agent, obs_dict, opponents, bot))

    if not records:
        raise RuntimeError("No ranker states collected")

    regrets = [record["regret_winrate"] for record in records]
    summary = {
        "states": len(records),
        "rows": int(sum(len(record["actions"]) for record in records)),
        "category_counts": dict(category_counts),
        "config_counts": dict(config_counts),
        "pair_counts": {
            f"{category}|{config}": count for (category, config), count in sorted(pair_counts.items())
        },
        "mean_regret": float(np.mean(regrets)),
        "high_regret_rate_teacher_margin": float(np.mean([r >= args.teacher_margin for r in regrets])),
        "mean_regret_by_category": {
            category: float(np.mean(values)) for category, values in regret_by_category.items()
        },
        "teacher_change_by_category": dict(changed_by_category),
        "stage_weight": {
            "mean": float(np.mean(stage_weights)) if stage_weights else 1.0,
            "min": float(np.min(stage_weights)) if stage_weights else 1.0,
            "max": float(np.max(stage_weights)) if stage_weights else 1.0,
        },
    }
    return records, summary


def padded_tensors(records, device, max_actions: int):
    n = len(records)
    obs = np.stack([record["obs"] for record in records]).astype(np.float32)
    hidden = np.stack([record["hidden"] for record in records]).astype(np.float32)
    belief = np.stack([record["belief"] for record in records]).astype(np.float32)
    actions = np.zeros((n, max_actions), dtype=np.int64)
    targets = np.zeros((n, max_actions), dtype=np.float32)
    valid = np.zeros((n, max_actions), dtype=np.bool_)
    model_index = np.zeros(n, dtype=np.int64)
    best_index = np.zeros(n, dtype=np.int64)

    for i, record in enumerate(records):
        count = min(len(record["actions"]), max_actions)
        actions[i, :count] = np.asarray(record["actions"][:count], dtype=np.int64)
        targets[i, :count] = np.asarray(record["targets"][:count], dtype=np.float32)
        valid[i, :count] = True
        if record["model_action"] in record["actions"][:count]:
            model_index[i] = record["actions"][:count].index(record["model_action"])
        if record["best_action"] in record["actions"][:count]:
            best_index[i] = record["actions"][:count].index(record["best_action"])

    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32, device=device),
        "hidden": torch.as_tensor(hidden, dtype=torch.float32, device=device),
        "belief": torch.as_tensor(belief, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "targets": torch.as_tensor(targets, dtype=torch.float32, device=device),
        "valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
        "model_action": torch.as_tensor([record["model_action"] for record in records], dtype=torch.long, device=device),
        "heuristic_action": torch.as_tensor(
            [record["heuristic_action"] for record in records], dtype=torch.long, device=device
        ),
        "model_index": torch.as_tensor(model_index, dtype=torch.long, device=device),
        "best_index": torch.as_tensor(best_index, dtype=torch.long, device=device),
        "stage_weight": torch.as_tensor(
            [record.get("stage_weight", 1.0) for record in records],
            dtype=torch.float32,
            device=device,
        ),
    }


def score_candidates(ranker: PairwiseActionRanker, data: dict, idx: torch.Tensor) -> torch.Tensor:
    actions = data["actions"][idx]
    batch, num_actions = actions.shape
    obs = data["obs"][idx].unsqueeze(1).expand(batch, num_actions, -1).reshape(batch * num_actions, -1)
    hidden = data["hidden"][idx].unsqueeze(1).expand(batch, num_actions, -1).reshape(batch * num_actions, -1)
    belief = data["belief"][idx].unsqueeze(1).expand(batch, num_actions, -1, -1).reshape(batch * num_actions, 3, 10)
    candidate = actions.reshape(-1)
    model_action = data["model_action"][idx].unsqueeze(1).expand(batch, num_actions).reshape(-1)
    heuristic_action = data["heuristic_action"][idx].unsqueeze(1).expand(batch, num_actions).reshape(-1)
    scores = ranker(obs, hidden, belief, candidate, model_action, heuristic_action)
    return scores.reshape(batch, num_actions)


def evaluate_ranker(ranker: PairwiseActionRanker, data: dict, idx: torch.Tensor, margin: float):
    ranker.eval()
    with torch.no_grad():
        scores = score_candidates(ranker, data, idx)
        scores = scores.masked_fill(~data["valid"][idx], -1e9)
        targets = data["targets"][idx].masked_fill(~data["valid"][idx], -1e9)
        pred_idx = scores.argmax(dim=-1)
        best_idx = targets.argmax(dim=-1)
        model_idx = data["model_index"][idx]
        pred_score = scores.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
        model_score = scores.gather(1, model_idx.unsqueeze(1)).squeeze(1)
        pred_override = (pred_idx != model_idx) & ((pred_score - model_score) >= margin)
        target_gap = targets.gather(1, best_idx.unsqueeze(1)).squeeze(1) - targets.gather(
            1, model_idx.unsqueeze(1)
        ).squeeze(1)
        target_changed = (best_idx != model_idx) & (target_gap >= margin)
        centered_pred = scores - scores.gather(1, model_idx.unsqueeze(1))
        centered_target = targets - targets.gather(1, model_idx.unsqueeze(1))
        valid = data["valid"][idx]
        mse = ((centered_pred[valid] - centered_target[valid]) ** 2).mean()
        mae = torch.abs(centered_pred[valid] - centered_target[valid]).mean()
        metrics = {
            "top1_acc": float((pred_idx == best_idx).float().mean().item()),
            "model_kept_rate": float((pred_idx == model_idx).float().mean().item()),
            "pred_override_rate": float(pred_override.float().mean().item()),
            "target_override_rate": float(target_changed.float().mean().item()),
            "override_agreement": float((pred_override == target_changed).float().mean().item()),
            "centered_mse": float(mse.item()),
            "centered_mae": float(mae.item()),
        }
    ranker.train()
    return metrics


def listwise_loss(scores, targets, valid, temperature, state_weight=None):
    masked_scores = scores.masked_fill(~valid, -1e9)
    masked_targets = targets.masked_fill(~valid, -1e9)
    target_dist = torch.softmax(masked_targets / temperature, dim=-1)
    log_probs = torch.log_softmax(masked_scores, dim=-1)
    loss = -(target_dist * log_probs).sum(dim=-1)
    if state_weight is not None:
        loss = loss * state_weight
        return loss.sum() / state_weight.sum().clamp_min(1e-6)
    return loss.mean()


def pairwise_loss(scores, targets, valid, min_gap, state_weight=None):
    diffs_target = targets.unsqueeze(2) - targets.unsqueeze(1)
    diffs_pred = scores.unsqueeze(2) - scores.unsqueeze(1)
    valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1) & (diffs_target >= min_gap)
    if not valid_pairs.any():
        return scores.new_tensor(0.0)
    weights = diffs_target[valid_pairs].clamp(max=1.0)
    if state_weight is not None:
        pair_state_weight = state_weight.view(-1, 1, 1).expand_as(diffs_target)[valid_pairs]
        weights = weights * pair_state_weight
    return (torch.nn.functional.softplus(-diffs_pred[valid_pairs]) * weights).mean()


def train_ranker(records, args, logger):
    device = torch.device(args.device)
    data = padded_tensors(records, device, args.max_actions)
    ranker = PairwiseActionRanker(args.hidden_dim, args.embed_dim).to(device)

    n = len(records)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 411)
    perm = torch.randperm(n, generator=generator, device=device)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    history = []

    logger.write(
        "Debut entrainement ranker Step3",
        expected="Apprendre un ordre de preference par action candidate, pas un label unique.",
        actual=f"train_states={len(train_idx)}, val_states={len(val_idx)}",
        details={"initial": evaluate_ranker(ranker, data, perm, args.eval_margin)},
    )

    for epoch in range(1, args.epochs + 1):
        losses = []
        for (idx,) in loader:
            scores = score_candidates(ranker, data, idx)
            valid = data["valid"][idx]
            targets = data["targets"][idx]
            state_weight = data["stage_weight"][idx]
            model_idx = data["model_index"][idx]
            centered_scores = scores - scores.gather(1, model_idx.unsqueeze(1))
            centered_targets = targets - targets.gather(1, model_idx.unsqueeze(1))
            row_weight = valid.float() * state_weight.unsqueeze(1)
            mse = (((centered_scores - centered_targets) ** 2) * row_weight).sum() / row_weight.sum().clamp_min(1e-6)
            lw = listwise_loss(scores, targets, valid, args.temperature, state_weight)
            pw = pairwise_loss(scores, targets, valid, args.pairwise_min_gap, state_weight)
            loss = args.mse_weight * mse + args.listwise_weight * lw + args.pairwise_weight * pw

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ranker.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": evaluate_ranker(ranker, data, train_idx, args.eval_margin),
            "val": evaluate_ranker(ranker, data, val_idx, args.eval_margin),
            "all": evaluate_ranker(ranker, data, perm, args.eval_margin),
        }
        history.append(row)
        logger.write(
            f"Epoch ranker {epoch}/{args.epochs}",
            expected="La val top1/MAE doit progresser avec un override rate controle.",
            actual=(
                f"loss={row['loss']:.4f}, val_top1={row['val']['top1_acc']:.3f}, "
                f"val_mae={row['val']['centered_mae']:.3f}, "
                f"val_override={row['val']['pred_override_rate']:.3f}"
            ),
            details=row,
        )

    return ranker, history


def jsonable_records(records, limit=50):
    output = []
    for record in records[:limit]:
        compact = {
            key: value
            for key, value in record.items()
            if key not in {"obs", "hidden", "belief"}
        }
        compact["model_decoded"] = decode_action(compact["model_action"])
        compact["heuristic_decoded"] = decode_action(compact["heuristic_action"])
        compact["best_decoded"] = decode_action(compact["best_action"])
        output.append(compact)
    return output


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train a Step3 pairwise action-value ranker.")
    parser.add_argument("--start", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--output", default="step3_pairwise_ranker_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=7000)
    parser.add_argument("--states-per-category", type=int, default=120)
    parser.add_argument(
        "--states-per-category-config",
        type=int,
        default=0,
        help="If >0, collect this many states for every category/opponent-config pair.",
    )
    parser.add_argument("--categories", nargs="+", default=["guard", "baron", "prince"])
    parser.add_argument("--collect-configs", nargs="+", default=list(OPPONENT_CONFIGS.keys()))
    parser.add_argument("--rollouts-per-action", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--reward-score-weight", type=float, default=0.05)
    parser.add_argument("--teacher-margin", type=float, default=0.10)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--stage-weight", type=float, default=0.0)
    parser.add_argument("--active-weight", type=float, default=0.0)
    parser.add_argument("--played-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-min-gap", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--eval-margin", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--log-every-states", type=int, default=40)
    parser.add_argument("--report", default="step3_pairwise_ranker_attempt1_train.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_pairwise_ranker_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=11200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.start)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step3 pairwise ranker",
        expected=(
            "Reprendre Step3 proprement: teacher rollout-guided, dataset plus large, "
            "apprentissage Q/ranking et evaluation rapide sans rollouts."
        ),
        actual=f"start={checkpoint}",
        details=vars(args),
    )

    records, collection_summary = collect_ranker_states(args, checkpoint, logger)
    logger.write(
        "Dataset ranker collecte",
        expected="Avoir assez d'etats values pour apprendre les deltas tactiques.",
        actual=f"states={collection_summary['states']}, rows={collection_summary['rows']}",
        details=collection_summary,
    )
    ranker, history = train_ranker(records, args, logger)

    output = Path(args.output)
    if output.parent == Path("."):
        output = CHECKPOINT_DIR / output
    payload = {
        "model_type": "step3_pairwise_action_ranker_v1",
        "created_at": now_stamp(),
        "base_checkpoint": str(checkpoint),
        "ranker": ranker.cpu().state_dict(),
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "categories": args.categories,
        "max_actions": args.max_actions,
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
        "Fin Step3 pairwise ranker",
        expected="Sauvegarder un candidat Step3 rapide a evaluer en arena.",
        actual=f"checkpoint={output}, report={report_path}",
        details={"collection_summary": collection_summary, "final_metrics": history[-1] if history else None},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
