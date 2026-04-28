"""Train Step3 v2: a CRN-filtered advantage head.

This version follows the expert feedback:

- compare candidate actions with paired/common-random-number rollouts;
- learn advantages relative to the Step2 action, not absolute Q values;
- mask noisy labels and use them only as weak tie regularization;
- keep enough metadata to audit whether the teacher signal is usable.
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
from step2_rl_finetune.evaluate_step2 import ModelSeat, OPPONENT_CONFIGS, random_action
from step3_action_value.mini_rollout_probe import (
    classify_state,
    decode_action,
    determinize_for_player,
)
from step3_action_value.common import (
    Player0FeaturePolicy,
    _debug_belief_array,
    action_parts,
    candidate_actions,
    opponent_action,
)


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def bounded_seed(value: int) -> int:
    return int(value % (2**32 - 1))


def rollout_once_paired(
    base_env,
    first_action,
    determinization_seed,
    playout_seed,
    checkpoint,
    opponents,
    continuation,
    heuristic_shuffle_targets=False,
):
    """Run one paired rollout.

    All candidate actions in the same rollout index receive the same sampled
    hidden world and the same stochastic playout seed. That makes the action
    difference much less noisy than independent rollouts.
    """

    env = determinize_for_player(base_env, "player_0", bounded_seed(determinization_seed))
    np.random.seed(bounded_seed(playout_seed))
    bot = HeuristicBot(shuffle_targets=heuristic_shuffle_targets)
    model = ModelSeat(checkpoint) if continuation == "model" else None
    reward0 = 0.0

    obs_dict, reward, terminated, truncated, _info = env.last()
    if env.agent_selection == "player_0":
        reward0 += float(reward)
    if terminated or truncated:
        env.step(None)
    else:
        env.step(int(first_action))

    for agent in env.agent_iter():
        obs_dict, reward, terminated, truncated, _info = env.last()
        if agent == "player_0":
            reward0 += float(reward)
        if terminated or truncated:
            env.step(None)
            continue
        if agent == "player_0":
            if continuation == "heuristic":
                action = bot.choose_action(env, agent)
            elif continuation == "model":
                action = model.act(obs_dict, agent)
            elif continuation == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(continuation)
        else:
            action = opponent_action(env, agent, obs_dict, opponents, bot)
        env.step(action)

    return float(reward0), int(reward0 >= 1.0)


def paired_correlation(score_matrix: np.ndarray) -> float | None:
    """Mean action-action correlation across paired rollout scores."""

    corrs = []
    for i in range(score_matrix.shape[0]):
        for j in range(i + 1, score_matrix.shape[0]):
            a = score_matrix[i]
            b = score_matrix[j]
            if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(corr):
                corrs.append(corr)
    return float(np.mean(corrs)) if corrs else None


def evaluate_candidate_actions_paired(env, actions, checkpoint, opponents, args, decision_seed):
    rewards_by_action = {int(action): [] for action in actions}
    wins_by_action = {int(action): [] for action in actions}

    for rollout in range(args.rollouts_per_action):
        det_seed = decision_seed * 1009 + rollout * 9173 + 17
        playout_seed = decision_seed * 7919 + rollout * 3571 + 29
        for action in actions:
            reward, win = rollout_once_paired(
                env,
                int(action),
                det_seed,
                playout_seed,
                checkpoint,
                opponents,
                args.player0_continuation,
                getattr(args, "heuristic_shuffle_targets", False),
            )
            rewards_by_action[int(action)].append(reward)
            wins_by_action[int(action)].append(win)

    rows = []
    score_matrix = []
    for action in actions:
        action = int(action)
        rewards = np.asarray(rewards_by_action[action], dtype=np.float32)
        wins = np.asarray(wins_by_action[action], dtype=np.float32)
        scores = wins + args.reward_score_weight * rewards
        score_matrix.append(scores)
        rows.append(
            {
                "action": action,
                "decoded": decode_action(action),
                "winrate": float(wins.mean()) if len(wins) else 0.0,
                "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
                "reward_std": float(rewards.std()) if len(rewards) else 0.0,
                "score": float(scores.mean()) if len(scores) else 0.0,
                "wins": int(wins.sum()),
                "rollouts": int(len(wins)),
                "_wins_array": wins,
                "_rewards_array": rewards,
                "_score_array": scores,
            }
        )
    rows.sort(key=lambda row: (row["score"], row["winrate"], row["mean_reward"]), reverse=True)
    corr = paired_correlation(np.stack(score_matrix, axis=0)) if len(score_matrix) > 1 else None
    return rows, corr


def paired_delta_stats(action_row, model_row) -> dict:
    win_diff = action_row["_wins_array"] - model_row["_wins_array"]
    score_diff = action_row["_score_array"] - model_row["_score_array"]
    n = int(len(score_diff))
    mean_win = float(win_diff.mean()) if n else 0.0
    mean_score = float(score_diff.mean()) if n else 0.0
    std = float(score_diff.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    if sem <= 1e-8:
        t_stat = float("inf") if abs(mean_score) > 1e-8 else 0.0
    else:
        t_stat = float(mean_score / sem)
    return {
        "n": n,
        "mean_win_delta": mean_win,
        "mean_score_delta": mean_score,
        "std_score_delta": std,
        "sem_score_delta": sem,
        "t_stat": t_stat,
    }


def belief_entropy(belief: np.ndarray) -> float:
    belief = np.asarray(belief, dtype=np.float32)
    if belief.size == 0:
        return 1.0
    probs = np.clip(belief, 1e-8, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=-1) / np.log(probs.shape[-1])
    return float(np.mean(entropy))


def state_features(env, belief: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    active_players = sum(1 for agent in env.possible_agents if not env.terminations.get(agent, False))
    played_cards = sum(len(cards) for cards in env._played_cards.values())
    deck_progress = 1.0 - (len(env._deck) / 21.0)
    return np.asarray(
        [
            deck_progress,
            min(played_cards / 16.0, 1.0),
            active_players / 4.0,
            (4 - active_players) / 3.0,
            belief_entropy(belief),
            min(float(action_mask.sum()) / 30.0, 1.0),
        ],
        dtype=np.float32,
    )


class AdvantageHeadV2(nn.Module):
    def __init__(self, hidden_dim: int = 256, embed_dim: int = 24, extra_dim: int = 6):
        super().__init__()
        self.card_emb = nn.Embedding(11, embed_dim)
        self.target_emb = nn.Embedding(10, embed_dim // 2)
        self.guess_emb = nn.Embedding(10, embed_dim // 2)
        action_dim = embed_dim + embed_dim // 2 + embed_dim // 2
        flag_dim = 4
        input_dim = OBS_DIM + LATENT + BELIEF_DIM + extra_dim + action_dim * 3 + flag_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.06),
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
        extra: torch.Tensor,
        candidate_action: torch.Tensor,
        model_action: torch.Tensor,
        heuristic_action: torch.Tensor,
    ) -> torch.Tensor:
        candidate_emb = self._embed_action(candidate_action)
        model_emb = self._embed_action(model_action)
        heuristic_emb = self._embed_action(heuristic_action)
        cand_card = torch.div(candidate_action, 100, rounding_mode="floor")
        model_card = torch.div(model_action, 100, rounding_mode="floor")
        heuristic_card = torch.div(heuristic_action, 100, rounding_mode="floor")
        flags = torch.stack(
            [
                (candidate_action == model_action).float(),
                (candidate_action == heuristic_action).float(),
                (cand_card == model_card).float(),
                (cand_card == heuristic_card).float(),
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                obs,
                hidden,
                belief.reshape(hidden.shape[0], -1),
                extra,
                candidate_emb,
                model_emb,
                heuristic_emb,
                flags,
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def collect_advantage_records(args, checkpoint: Path, logger: ExperimentLogger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    category_counts = Counter()
    config_counts = Counter()
    pair_counts = Counter()
    significant_by_category = Counter()
    positive_override_by_category = Counter()
    crn_correlations = []
    target_per_pair = args.states_per_category_config

    def target_reached() -> bool:
        if target_per_pair:
            return all(
                pair_counts[(category, config)] >= target_per_pair
                for category in args.categories
                for config in args.collect_configs
            )
        return all(category_counts[category] >= args.states_per_category for category in args.categories)

    def can_collect(category: str, config: str) -> bool:
        if target_per_pair:
            return pair_counts[(category, config)] < target_per_pair
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
                    paired_args = argparse.Namespace(
                        rollouts_per_action=args.rollouts_per_action,
                        player0_continuation=args.player0_continuation,
                        reward_score_weight=args.reward_score_weight,
                    )
                    rows, corr = evaluate_candidate_actions_paired(
                        env,
                        candidates,
                        checkpoint,
                        opponents,
                        paired_args,
                        decision_seed=seed * 100 + turn,
                    )
                    if corr is not None:
                        crn_correlations.append(corr)

                    by_action = {row["action"]: row for row in rows}
                    model_row = by_action.get(model_action)
                    if model_row is None:
                        env.step(model_action)
                        continue

                    actions = []
                    targets = []
                    label_mask = []
                    weights = []
                    stats = []
                    state_significant = 0
                    best_significant_action = int(model_action)
                    best_significant_advantage = 0.0
                    for row in rows:
                        action = int(row["action"])
                        delta = paired_delta_stats(row, model_row)
                        is_model = action == int(model_action)
                        significant = (
                            is_model
                            or (
                                abs(delta["mean_win_delta"]) >= args.min_win_delta
                                and abs(delta["t_stat"]) >= args.t_threshold
                            )
                        )
                        if significant and not is_model:
                            state_significant += 1
                        target = float(delta["mean_score_delta"])
                        if is_model:
                            target = 0.0
                            weight = 1.0
                        elif significant:
                            confidence = min(args.max_confidence_weight, max(1.0, abs(delta["t_stat"]) / args.t_threshold))
                            effect = min(args.max_confidence_weight, max(1.0, abs(delta["mean_win_delta"]) / args.min_win_delta))
                            weight = float(confidence * effect)
                        else:
                            target = 0.0
                            weight = args.tie_weight
                        if significant and target > best_significant_advantage:
                            best_significant_advantage = target
                            best_significant_action = action
                        actions.append(action)
                        targets.append(target)
                        label_mask.append(bool(significant))
                        weights.append(float(weight))
                        clean = {key: value for key, value in row.items() if not key.startswith("_")}
                        clean.update(delta)
                        clean["significant_vs_model"] = bool(significant and not is_model)
                        clean["target_advantage"] = float(target)
                        clean["label_weight"] = float(weight)
                        stats.append(clean)

                    category_counts[category] += 1
                    config_counts[config_name] += 1
                    pair_counts[(category, config_name)] += 1
                    significant_by_category[category] += state_significant
                    if best_significant_action != int(model_action) and best_significant_advantage >= args.min_positive_override:
                        positive_override_by_category[category] += 1

                    records.append(
                        {
                            "category": category,
                            "config": config_name,
                            "seed": seed,
                            "turn": turn,
                            "obs": obs_dict["observation"].astype(np.float32),
                            "hidden": hidden.astype(np.float32),
                            "belief": belief.astype(np.float32),
                            "extra": state_features(env, belief, obs_dict["action_mask"]),
                            "model_action": int(model_action),
                            "heuristic_action": int(heuristic_action),
                            "actions": actions,
                            "targets": targets,
                            "label_mask": label_mask,
                            "weights": weights,
                            "best_significant_action": int(best_significant_action),
                            "best_significant_advantage": float(best_significant_advantage),
                            "significant_actions": int(state_significant),
                            "crn_correlation": corr,
                            "deck_size": int(len(env._deck)),
                            "active_players": int(
                                sum(1 for player in env.possible_agents if not env.terminations.get(player, False))
                            ),
                            "played_cards_count": int(sum(len(cards) for cards in env._played_cards.values())),
                            "top_actions": stats[: min(8, len(stats))],
                        }
                    )

                    if len(records) % args.log_every_states == 0:
                        logger.write(
                            "Collecte Step3 v2 advantage",
                            expected="Labels CRN apparies avec filtrage statistique, pas de preferences faibles forcees.",
                            actual=(
                                f"states={len(records)}, categories={dict(category_counts)}, "
                                f"mean_crn_corr={np.mean(crn_correlations) if crn_correlations else None}"
                            ),
                            details={
                                "config_counts": dict(config_counts),
                                "significant_by_category": dict(significant_by_category),
                                "positive_override_by_category": dict(positive_override_by_category),
                            },
                        )
                env.step(model_action)
            else:
                env.step(opponent_action(env, agent, obs_dict, opponents, bot))

    if not records:
        raise RuntimeError("No Step3 v2 records collected")

    non_model_rows = sum(max(0, len(record["actions"]) - 1) for record in records)
    significant_rows = sum(record["significant_actions"] for record in records)
    positive_states = sum(
        1
        for record in records
        if record["best_significant_action"] != record["model_action"]
        and record["best_significant_advantage"] >= args.min_positive_override
    )
    advantages = [
        record["best_significant_advantage"]
        for record in records
        if record["best_significant_action"] != record["model_action"]
    ]
    summary = {
        "states": len(records),
        "rows": int(sum(len(record["actions"]) for record in records)),
        "category_counts": dict(category_counts),
        "config_counts": dict(config_counts),
        "pair_counts": {
            f"{category}|{config}": count for (category, config), count in sorted(pair_counts.items())
        },
        "significant_action_rows": int(significant_rows),
        "non_model_action_rows": int(non_model_rows),
        "significant_action_rate": float(significant_rows / max(1, non_model_rows)),
        "positive_override_states": int(positive_states),
        "positive_override_state_rate": float(positive_states / len(records)),
        "mean_positive_advantage": float(np.mean(advantages)) if advantages else 0.0,
        "crn_correlation": {
            "mean": float(np.mean(crn_correlations)) if crn_correlations else None,
            "count": int(len(crn_correlations)),
        },
        "significant_by_category": dict(significant_by_category),
        "positive_override_by_category": dict(positive_override_by_category),
    }
    return records, summary


def padded_tensors(records, device, max_actions: int):
    n = len(records)
    obs = np.stack([record["obs"] for record in records]).astype(np.float32)
    hidden = np.stack([record["hidden"] for record in records]).astype(np.float32)
    belief = np.stack([record["belief"] for record in records]).astype(np.float32)
    extra = np.stack([record["extra"] for record in records]).astype(np.float32)
    actions = np.zeros((n, max_actions), dtype=np.int64)
    targets = np.zeros((n, max_actions), dtype=np.float32)
    label_mask = np.zeros((n, max_actions), dtype=np.bool_)
    valid = np.zeros((n, max_actions), dtype=np.bool_)
    weights = np.zeros((n, max_actions), dtype=np.float32)
    model_index = np.zeros(n, dtype=np.int64)

    for i, record in enumerate(records):
        count = min(len(record["actions"]), max_actions)
        actions[i, :count] = np.asarray(record["actions"][:count], dtype=np.int64)
        targets[i, :count] = np.asarray(record["targets"][:count], dtype=np.float32)
        label_mask[i, :count] = np.asarray(record["label_mask"][:count], dtype=np.bool_)
        weights[i, :count] = np.asarray(record["weights"][:count], dtype=np.float32)
        valid[i, :count] = True
        if record["model_action"] in record["actions"][:count]:
            model_index[i] = record["actions"][:count].index(record["model_action"])

    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32, device=device),
        "hidden": torch.as_tensor(hidden, dtype=torch.float32, device=device),
        "belief": torch.as_tensor(belief, dtype=torch.float32, device=device),
        "extra": torch.as_tensor(extra, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "targets": torch.as_tensor(targets, dtype=torch.float32, device=device),
        "label_mask": torch.as_tensor(label_mask, dtype=torch.bool, device=device),
        "valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
        "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
        "model_action": torch.as_tensor([record["model_action"] for record in records], dtype=torch.long, device=device),
        "heuristic_action": torch.as_tensor(
            [record["heuristic_action"] for record in records], dtype=torch.long, device=device
        ),
        "model_index": torch.as_tensor(model_index, dtype=torch.long, device=device),
    }


def score_candidates(head: AdvantageHeadV2, data: dict, idx: torch.Tensor) -> torch.Tensor:
    actions = data["actions"][idx]
    batch, num_actions = actions.shape
    obs = data["obs"][idx].unsqueeze(1).expand(batch, num_actions, -1).reshape(batch * num_actions, -1)
    hidden = data["hidden"][idx].unsqueeze(1).expand(batch, num_actions, -1).reshape(batch * num_actions, -1)
    belief = data["belief"][idx].unsqueeze(1).expand(batch, num_actions, -1, -1).reshape(batch * num_actions, 3, 10)
    extra = data["extra"][idx].unsqueeze(1).expand(batch, num_actions, -1).reshape(batch * num_actions, -1)
    candidate = actions.reshape(-1)
    model_action = data["model_action"][idx].unsqueeze(1).expand(batch, num_actions).reshape(-1)
    heuristic_action = data["heuristic_action"][idx].unsqueeze(1).expand(batch, num_actions).reshape(-1)
    scores = head(obs, hidden, belief, extra, candidate, model_action, heuristic_action)
    scores = scores.reshape(batch, num_actions)
    model_idx = data["model_index"][idx]
    return scores - scores.gather(1, model_idx.unsqueeze(1))


def masked_huber(pred, target, weights, mask):
    if not mask.any():
        return pred.new_tensor(0.0)
    loss = torch.nn.functional.smooth_l1_loss(pred[mask], target[mask], reduction="none")
    w = weights[mask]
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def pairwise_advantage_loss(scores, targets, label_mask, weights, min_gap, beta):
    diffs_target = targets.unsqueeze(2) - targets.unsqueeze(1)
    diffs_pred = scores.unsqueeze(2) - scores.unsqueeze(1)
    valid_pairs = label_mask.unsqueeze(2) & label_mask.unsqueeze(1) & (diffs_target >= min_gap)
    if not valid_pairs.any():
        return scores.new_tensor(0.0)
    pair_weights = (weights.unsqueeze(2) + weights.unsqueeze(1)) * 0.5
    pair_weights = pair_weights[valid_pairs].clamp_min(1e-4)
    scaled = diffs_pred[valid_pairs] / beta
    return (torch.nn.functional.softplus(-scaled) * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)


def step2_trust_region_kl(scores, targets, valid, model_index, args):
    """KL regularizer toward the Step2 no-override reference policy.

    The advantage head is not a full actor, but at inference it becomes a
    categorical choice over candidate actions plus a margin gate. This KL keeps
    the candidate distribution close to "choose Step2" unless the CRN teacher
    found a large positive advantage for another action.
    """

    weight = float(getattr(args, "trust_region_kl_weight", 0.0))
    if weight <= 0.0:
        return scores.new_tensor(0.0)

    temperature = max(1e-6, float(getattr(args, "trust_region_temperature", 0.25)))
    epsilon = max(0.0, min(0.49, float(getattr(args, "trust_region_step2_epsilon", 0.02))))
    break_advantage = float(getattr(args, "trust_region_break_advantage", 0.20))
    break_weight = max(0.0, float(getattr(args, "trust_region_break_weight", 0.15)))

    masked_scores = (scores / temperature).masked_fill(~valid, -1e9)
    log_probs = torch.nn.functional.log_softmax(masked_scores, dim=-1)
    valid_count = valid.sum(dim=-1).float()
    eps = torch.where(valid_count > 1, scores.new_full(valid_count.shape, epsilon), scores.new_zeros(valid_count.shape))
    non_model = (valid_count - 1.0).clamp_min(1.0)
    ref_probs = valid.float() * (eps / non_model).unsqueeze(1)
    ref_probs.scatter_(1, model_index.unsqueeze(1), (1.0 - eps).unsqueeze(1))

    target_values = targets.masked_fill(~valid, -1e9)
    target_idx = target_values.argmax(dim=-1)
    target_score = target_values.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    target_override = (target_idx != model_index) & (target_score >= break_advantage)
    state_weight = torch.where(
        target_override,
        scores.new_full(target_override.shape, break_weight),
        scores.new_ones(target_override.shape),
    )

    kl = (ref_probs * (torch.log(ref_probs.clamp_min(1e-8)) - log_probs)).sum(dim=-1)
    return (kl * state_weight).mean()


def evaluate_advantage_head(head: AdvantageHeadV2, data: dict, idx: torch.Tensor, margin: float):
    head.eval()
    with torch.no_grad():
        scores = score_candidates(head, data, idx)
        scores = scores.masked_fill(~data["valid"][idx], -1e9)
        targets = data["targets"][idx].masked_fill(~data["valid"][idx], -1e9)
        label_mask = data["label_mask"][idx]
        valid = data["valid"][idx]
        model_idx = data["model_index"][idx]
        pred_idx = scores.argmax(dim=-1)
        pred_score = scores.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
        pred_override = (pred_idx != model_idx) & (pred_score >= margin)
        target_idx = targets.argmax(dim=-1)
        target_score = targets.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        target_override = (target_idx != model_idx) & (target_score >= margin)
        train_mask = label_mask | ((~label_mask) & valid)
        mae = torch.abs(scores[train_mask] - data["targets"][idx][train_mask]).mean()
        significant_mae = (
            torch.abs(scores[label_mask] - data["targets"][idx][label_mask]).mean()
            if label_mask.any()
            else scores.new_tensor(0.0)
        )
        metrics = {
            "top1_acc": float((pred_idx == target_idx).float().mean().item()),
            "pred_override_rate": float(pred_override.float().mean().item()),
            "target_override_rate": float(target_override.float().mean().item()),
            "override_agreement": float((pred_override == target_override).float().mean().item()),
            "centered_mae_all": float(mae.item()),
            "centered_mae_significant": float(significant_mae.item()),
            "mean_pred_margin": float(pred_score.mean().item()),
            "mean_target_margin": float(target_score.mean().item()),
        }
    head.train()
    return metrics


def train_head(records, args, logger):
    device = torch.device(args.device)
    data = padded_tensors(records, device, args.max_actions)
    head = AdvantageHeadV2(args.hidden_dim, args.embed_dim).to(device)

    init_head_checkpoint = getattr(args, "init_head_checkpoint", None)
    if init_head_checkpoint:
        init_path = Path(init_head_checkpoint)
        candidates = [init_path, CHECKPOINT_DIR / init_path, PROJECT_ROOT / init_path]
        for candidate in candidates:
            if candidate.exists():
                init_path = candidate
                break
        else:
            raise FileNotFoundError(f"Initial advantage checkpoint not found: {init_head_checkpoint}")
        init_payload = torch.load(init_path, map_location=device, weights_only=True)
        if init_payload.get("model_type") != "step3_advantage_head_v2":
            raise ValueError(f"{init_path} is not a Step3 v2 advantage checkpoint")
        head.load_state_dict(init_payload["head"])
        logger.write(
            "Initialisation Step3 v2 advantage",
            expected="DAgger doit partir de la tete rapide precedente plutot que reoublier les acquis.",
            actual=f"init_head_checkpoint={init_path}",
            details={
                "source_created_at": init_payload.get("created_at"),
                "source_base_checkpoint": init_payload.get("base_checkpoint"),
                "source_collection": init_payload.get("metadata", {}).get("collection_summary"),
            },
        )

    n = len(records)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 811)
    perm = torch.randperm(n, generator=generator, device=device)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    history = []

    logger.write(
        "Debut entrainement Step3 v2 advantage",
        expected="Apprendre des avantages relatifs calibres avec peu de gradients sur le bruit.",
        actual=f"train_states={len(train_idx)}, val_states={len(val_idx)}",
        details={"initial": evaluate_advantage_head(head, data, perm, args.eval_margin)},
    )

    for epoch in range(1, args.epochs + 1):
        losses = []
        trust_losses = []
        for (idx,) in loader:
            scores = score_candidates(head, data, idx)
            targets = data["targets"][idx]
            label_mask = data["label_mask"][idx]
            valid = data["valid"][idx]
            weights = data["weights"][idx]
            tie_mask = valid & ~label_mask & (weights > 0.0)
            supervised = masked_huber(scores, targets, weights, label_mask)
            tie = masked_huber(scores, targets, weights, tie_mask)
            pairwise = pairwise_advantage_loss(
                scores,
                targets,
                label_mask,
                weights,
                args.pairwise_min_gap,
                args.pairwise_beta,
            )
            model_zero = (scores.gather(1, data["model_index"][idx].unsqueeze(1)).squeeze(1) ** 2).mean()
            trust_kl = step2_trust_region_kl(
                scores,
                targets,
                valid,
                data["model_index"][idx],
                args,
            )
            loss = (
                args.supervised_weight * supervised
                + args.tie_loss_weight * tie
                + args.pairwise_weight * pairwise
                + args.model_zero_weight * model_zero
                + getattr(args, "trust_region_kl_weight", 0.0) * trust_kl
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            trust_losses.append(float(trust_kl.item()))

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "trust_region_kl": float(np.mean(trust_losses)) if trust_losses else 0.0,
            "train": evaluate_advantage_head(head, data, train_idx, args.eval_margin),
            "val": evaluate_advantage_head(head, data, val_idx, args.eval_margin),
            "all": evaluate_advantage_head(head, data, perm, args.eval_margin),
        }
        history.append(row)
        logger.write(
            f"Epoch Step3 v2 advantage {epoch}/{args.epochs}",
            expected="La val doit ameliorer le top1/MAE sans exploser le taux d'override.",
            actual=(
                f"loss={row['loss']:.4f}, val_top1={row['val']['top1_acc']:.3f}, "
                f"val_mae_sig={row['val']['centered_mae_significant']:.3f}, "
                f"val_override={row['val']['pred_override_rate']:.3f}, "
                f"trust_kl={row['trust_region_kl']:.4f}"
            ),
            details=row,
        )

    return head, history


def jsonable_records(records, limit=60):
    output = []
    for record in records[:limit]:
        compact = {key: value for key, value in record.items() if key not in {"obs", "hidden", "belief", "extra"}}
        compact["model_decoded"] = decode_action(compact["model_action"])
        compact["heuristic_decoded"] = decode_action(compact["heuristic_action"])
        compact["best_significant_decoded"] = decode_action(compact["best_significant_action"])
        output.append(compact)
    return output


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train Step3 v2 CRN advantage head.")
    parser.add_argument("--start", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--output", default="step3_advantage_v2_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=9000)
    parser.add_argument("--states-per-category", type=int, default=140)
    parser.add_argument("--states-per-category-config", type=int, default=0)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["guard", "priest", "spy", "king", "prince", "chancellor_card", "baron"],
    )
    parser.add_argument("--collect-configs", nargs="+", default=list(OPPONENT_CONFIGS.keys()))
    parser.add_argument("--rollouts-per-action", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--reward-score-weight", type=float, default=0.05)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--min-win-delta", type=float, default=0.10)
    parser.add_argument("--min-positive-override", type=float, default=0.08)
    parser.add_argument("--t-threshold", type=float, default=1.35)
    parser.add_argument("--tie-weight", type=float, default=0.04)
    parser.add_argument("--max-confidence-weight", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--tie-loss-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.8)
    parser.add_argument("--pairwise-min-gap", type=float, default=0.08)
    parser.add_argument("--pairwise-beta", type=float, default=0.08)
    parser.add_argument("--model-zero-weight", type=float, default=0.02)
    parser.add_argument("--trust-region-kl-weight", type=float, default=0.0)
    parser.add_argument("--trust-region-temperature", type=float, default=0.25)
    parser.add_argument("--trust-region-step2-epsilon", type=float, default=0.02)
    parser.add_argument("--trust-region-break-advantage", type=float, default=0.20)
    parser.add_argument("--trust-region-break-weight", type=float, default=0.15)
    parser.add_argument("--eval-margin", type=float, default=0.08)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--log-every-states", type=int, default=40)
    parser.add_argument("--report", default="step3_advantage_v2_attempt1_train.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_advantage_v2_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13100)
    parser.add_argument("--init-head-checkpoint", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.start)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step3 v2 advantage",
        expected=(
            "Refaire Step3 proprement: CRN, avantage relatif a Step2, filtrage statistique, "
            "paires ambiguës seulement en regularisation faible."
        ),
        actual=f"start={checkpoint}",
        details=vars(args),
    )

    records, collection_summary = collect_advantage_records(args, checkpoint, logger)
    logger.write(
        "Dataset Step3 v2 collecte",
        expected="Avoir un taux significatif de paires fiables sans forcer le bruit.",
        actual=(
            f"states={collection_summary['states']}, "
            f"significant_rate={collection_summary['significant_action_rate']:.2%}, "
            f"positive_override_rate={collection_summary['positive_override_state_rate']:.2%}"
        ),
        details=collection_summary,
    )

    head, history = train_head(records, args, logger)

    output = Path(args.output)
    if output.parent == Path("."):
        output = CHECKPOINT_DIR / output
    payload = {
        "model_type": "step3_advantage_head_v2",
        "created_at": now_stamp(),
        "base_checkpoint": str(checkpoint),
        "head": head.cpu().state_dict(),
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "extra_dim": 6,
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
        "Fin Step3 v2 advantage",
        expected="Sauvegarder un candidat autonome a evaluer contre Step2.",
        actual=f"checkpoint={output}, report={report_path}",
        details={"collection_summary": collection_summary, "final_metrics": history[-1] if history else None},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
