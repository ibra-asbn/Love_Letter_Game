"""Seat-rotated fair arena with tactical diagnostics.

This script evaluates the current policies while rotating the evaluated player
through all four seats. Opponent heuristics use randomized target tie-breaks so
the benchmark is not a player_0 focus test.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import (
    absolute_to_relative_action,
    absolute_to_relative_mask,
    relative_to_absolute_action,
)
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import random_action, summarize_rewards
from step3_action_value.evaluate_advantage_head_v2 import (
    dynamic_margin,
    load_advantage_bundle,
    state_belief_entropy,
    summarize_advantage,
)
from step3_action_value.mini_rollout_probe import CARD_NAMES, classify_state, decode_action, determinize_for_player
from step3_action_value.train_advantage_head_v2 import paired_delta_stats, state_features
from step3_action_value.common import _debug_belief_array


INTERLUDE_DIR = PROJECT_ROOT / "interlude_heuristic_comparison"
REPORT_DIR = INTERLUDE_DIR / "reports"
LOG_DIR = INTERLUDE_DIR / "logs"

CONFIG_HEURISTIC_COUNTS = {
    "vs_0H_3R": 0,
    "vs_1H_2R": 1,
    "vs_2H_1R": 2,
    "vs_3H": 3,
}

CONFIG_LABELS = {
    "vs_0H_3R": "vs 3 randoms",
    "vs_1H_2R": "vs 1H+2R",
    "vs_2H_1R": "vs 2H+1R",
    "vs_3H": "vs 3H",
}

POLICY_LABELS = {
    "heuristic_fair": "Fair HeuristicBot",
    "step2_retarget": "Step2 retarget",
    "step3_fast_dagger": "Step3 rapide DAgger",
    "step3_hybrid_verify16": "Step3 hybride verify16",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def bounded_seed(value: int) -> int:
    return int(value % (2**32 - 1))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def known_card_for(env, observer: str, target: str) -> int | None:
    known = env._known_cards[observer][target]
    idx = np.where(known >= 1.0)[0]
    return int(idx[0]) if len(idx) else None


def relative_opponents(eval_agent: str) -> list[str]:
    agents = [f"player_{i}" for i in range(4)]
    idx = agents.index(eval_agent)
    return [agents[(idx + offset) % 4] for offset in range(1, 4)]


def build_roles(eval_agent: str, heuristic_count: int, game_index: int) -> dict[str, str]:
    roles = {agent: "model" for agent in [eval_agent]}
    opponents = relative_opponents(eval_agent)
    if heuristic_count == 0:
        heuristic_slots = set()
    elif heuristic_count == 3:
        heuristic_slots = {0, 1, 2}
    elif heuristic_count == 1:
        heuristic_slots = {(game_index // 4) % 3}
    elif heuristic_count == 2:
        combos = [(0, 1), (0, 2), (1, 2)]
        heuristic_slots = set(combos[(game_index // 4) % len(combos)])
    else:
        raise ValueError(heuristic_count)
    for slot, agent in enumerate(opponents):
        roles[agent] = "heuristic" if slot in heuristic_slots else "random"
    return roles


def role_action(env, agent: str, obs_dict, roles: dict[str, str], policies: dict[str, object], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "model":
        return policies[agent].act(env, obs_dict, agent)
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


class FairHeuristicSeat:
    def __init__(self):
        self.bot = HeuristicBot(shuffle_targets=True)

    def act(self, env, obs_dict, agent: str) -> int:
        return int(self.bot.choose_action(env, agent))


class RelativeModelSeat:
    def __init__(self, checkpoint: Path):
        self.policy = load_belief_policy(checkpoint)
        self.state = None

    def act(self, env, obs_dict, agent: str) -> int:
        my_idx = int(agent.rsplit("_", 1)[1])
        relative_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        relative_action, self.state = self.policy.act(relative_obs, self.state, agent_id=agent)
        return relative_to_absolute_action(int(relative_action), my_idx)


def generic_candidate_actions(env, agent: str, model_action: int, heuristic_action: int, max_actions: int) -> list[int]:
    valid = [int(action) for action in np.where(env.observe(agent)["action_mask"] == 1)[0]]
    if len(valid) <= max_actions:
        selected = valid
    else:
        by_card = defaultdict(list)
        for action in valid:
            card = 99 if action >= 900 else action // 100
            by_card[card].append(action)
        selected = []
        for card in sorted(by_card):
            selected.extend(by_card[card][: max(1, max_actions // max(1, len(by_card)))])
        for action in valid:
            if len(selected) >= max_actions:
                break
            if action not in selected:
                selected.append(action)
    for forced in [int(model_action), int(heuristic_action)]:
        if forced not in selected:
            selected = [forced] + selected
    return list(dict.fromkeys(selected))[:max_actions]


class GenericAdvantageSeat:
    def __init__(
        self,
        base_checkpoint: Path,
        head,
        categories: list[str],
        args,
        roles: dict[str, str],
        eval_agent: str,
    ):
        self.base_checkpoint = base_checkpoint
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.head = head.to(args.device).eval()
        self.categories = set(categories)
        self.max_actions = args.max_actions
        self.override_margin = args.override_margin
        self.entropy_margin_scale = 0.0
        self.verify_rollouts = args.verify_rollouts
        self.verify_min_win_delta = args.verify_min_win_delta
        self.verify_min_score_delta = args.verify_min_score_delta
        self.verify_t_threshold = args.verify_t_threshold
        self.verify_player_continuation = "heuristic"
        self.verify_reward_score_weight = 0.05
        self.device = torch.device(args.device)
        self.roles = roles
        self.eval_agent = eval_agent
        self.bot = HeuristicBot(shuffle_targets=True)
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)

    def _score(self, obs, hidden, belief, extra, actions, model_action, heuristic_action):
        n = len(actions)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1, -1)
        extra_t = torch.as_tensor(extra, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        model_t = torch.full_like(actions_t, int(model_action))
        heuristic_t = torch.full_like(actions_t, int(heuristic_action))
        with torch.no_grad():
            scores = self.head(obs_t, hidden_t, belief_t, extra_t, actions_t, model_t, heuristic_t)
        model_positions = [i for i, action in enumerate(actions) if int(action) == int(model_action)]
        if model_positions:
            scores = scores - scores[model_positions[0]]
        return scores

    def _verify_override(self, env, model_action_abs: int, best_action_abs: int, decision_seed: int):
        if self.verify_rollouts <= 0 or best_action_abs == model_action_abs:
            return True, None
        rows, corr = evaluate_candidate_actions_paired_for_agent(
            env=env,
            eval_agent=self.eval_agent,
            actions=[int(model_action_abs), int(best_action_abs)],
            checkpoint=self.base_checkpoint,
            roles=self.roles,
            rollouts_per_action=self.verify_rollouts,
            continuation=self.verify_player_continuation,
            reward_score_weight=self.verify_reward_score_weight,
            decision_seed=decision_seed,
        )
        by_action = {int(row["action"]): row for row in rows}
        model_row = by_action.get(int(model_action_abs))
        best_row = by_action.get(int(best_action_abs))
        if model_row is None or best_row is None:
            return False, {"missing_rows": True, "crn_correlation": corr}
        delta = paired_delta_stats(best_row, model_row)
        accepted = (
            delta["mean_win_delta"] >= self.verify_min_win_delta
            and delta["mean_score_delta"] >= self.verify_min_score_delta
            and delta["t_stat"] >= self.verify_t_threshold
        )
        return accepted, {"delta": delta, "crn_correlation": corr}

    def act(self, env, obs_dict, agent: str) -> int:
        my_idx = int(agent.rsplit("_", 1)[1])
        rel_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        model_action_rel, self.state = self.base.act(rel_obs, self.state, agent_id=agent)
        model_action_rel = int(model_action_rel)
        model_action_abs = relative_to_absolute_action(model_action_rel, my_idx)

        category = classify_state(env, agent)
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1
        if category not in self.categories or int(obs_dict["action_mask"].sum()) <= 1:
            return int(model_action_abs)

        heuristic_action_abs = int(self.bot.choose_action(env, agent))
        heuristic_action_rel = absolute_to_relative_action(heuristic_action_abs, my_idx)
        candidate_abs = generic_candidate_actions(env, agent, model_action_abs, heuristic_action_abs, self.max_actions)

        pairs = []
        seen_rel = set()
        for action_abs in candidate_abs:
            action_rel = absolute_to_relative_action(action_abs, my_idx)
            if action_rel in seen_rel:
                continue
            pairs.append((int(action_abs), int(action_rel)))
            seen_rel.add(action_rel)
        if model_action_rel not in seen_rel:
            pairs.insert(0, (int(model_action_abs), int(model_action_rel)))

        actions_abs = [pair[0] for pair in pairs]
        actions_rel = [pair[1] for pair in pairs]

        hidden = self.state.detach().cpu().squeeze(0).numpy().astype(np.float32)
        belief = _debug_belief_array(getattr(self.base, "last_debug", None))
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        belief = belief.astype(np.float32)
        extra = state_features(env, belief, rel_obs["action_mask"])
        scores = self._score(
            rel_obs["observation"].astype(np.float32),
            hidden,
            belief,
            extra,
            actions_rel,
            model_action_rel,
            heuristic_action_rel,
        )

        best_idx = int(torch.argmax(scores).item())
        best_action_abs = int(actions_abs[best_idx])
        best_score = float(scores[best_idx].item())
        margin = dynamic_margin(self.override_margin, self.entropy_margin_scale, belief)

        self.stats["advantage_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += best_score
        self.category_stats[category]["sum_required_margin"] += margin

        if best_action_abs != model_action_abs and best_score >= margin:
            decision_seed = bounded_seed(
                (my_idx + 1) * 1_000_003
                + int(self.stats["decisions"]) * 9176
                + len(env._deck) * 131
                + sum(len(cards) for cards in env._played_cards.values()) * 17
                + int(best_action_abs)
            )
            accepted, verify = self._verify_override(
                env,
                int(model_action_abs),
                int(best_action_abs),
                decision_seed=decision_seed,
            )
            if self.verify_rollouts > 0:
                self.stats["verify_checks"] += 1
                self.category_stats[category]["verify_checks"] += 1
                if verify and "delta" in verify:
                    delta = verify["delta"]
                    self.stats["sum_verify_win_delta"] += delta["mean_win_delta"]
                    self.stats["sum_verify_score_delta"] += delta["mean_score_delta"]
                    self.stats["sum_verify_t"] += delta["t_stat"]
                    self.category_stats[category]["sum_verify_win_delta"] += delta["mean_win_delta"]
                    self.category_stats[category]["sum_verify_score_delta"] += delta["mean_score_delta"]
                    self.category_stats[category]["sum_verify_t"] += delta["t_stat"]
                if accepted:
                    self.stats["verify_accepts"] += 1
                    self.category_stats[category]["verify_accepts"] += 1
                else:
                    self.stats["verify_rejects"] += 1
                    self.category_stats[category]["verify_rejects"] += 1
                    return int(model_action_abs)
            self.stats["overrides"] += 1
            self.stats["sum_override_margin"] += best_score
            self.category_stats[category]["overrides"] += 1
            self.category_stats[category]["sum_override_margin"] += best_score
            return int(best_action_abs)
        return int(model_action_abs)


def rollout_once_paired_for_agent(
    base_env,
    eval_agent: str,
    first_action: int,
    determinization_seed: int,
    playout_seed: int,
    checkpoint: Path,
    roles: dict[str, str],
    continuation: str,
) -> tuple[float, int]:
    env = determinize_for_player(base_env, eval_agent, bounded_seed(determinization_seed))
    np.random.seed(bounded_seed(playout_seed))
    bot = HeuristicBot(shuffle_targets=True)
    model = RelativeModelSeat(checkpoint) if continuation == "model" else None
    reward_eval = 0.0

    obs_dict, reward, terminated, truncated, _info = env.last()
    if env.agent_selection == eval_agent:
        reward_eval += float(reward)
    if terminated or truncated:
        env.step(None)
    else:
        env.step(int(first_action))

    for agent in env.agent_iter():
        obs_dict, reward, terminated, truncated, _info = env.last()
        if agent == eval_agent:
            reward_eval += float(reward)
        if terminated or truncated:
            env.step(None)
            continue
        if agent == eval_agent:
            if continuation == "heuristic":
                action = int(bot.choose_action(env, agent))
            elif continuation == "model":
                action = int(model.act(env, obs_dict, agent))
            elif continuation == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(continuation)
        else:
            role = roles[agent]
            if role == "heuristic":
                action = int(bot.choose_action(env, agent))
            elif role == "random":
                action = random_action(obs_dict)
            elif role == "model":
                action = random_action(obs_dict)
            else:
                raise ValueError(role)
        env.step(action)
    return float(reward_eval), int(reward_eval >= 1.0)


def paired_correlation(score_matrix: np.ndarray) -> float | None:
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


def evaluate_candidate_actions_paired_for_agent(
    env,
    eval_agent: str,
    actions: list[int],
    checkpoint: Path,
    roles: dict[str, str],
    rollouts_per_action: int,
    continuation: str,
    reward_score_weight: float,
    decision_seed: int,
):
    rewards_by_action = {int(action): [] for action in actions}
    wins_by_action = {int(action): [] for action in actions}

    for rollout in range(rollouts_per_action):
        det_seed = decision_seed * 1009 + rollout * 9173 + 17
        playout_seed = decision_seed * 7919 + rollout * 3571 + 29
        for action in actions:
            reward, win = rollout_once_paired_for_agent(
                env,
                eval_agent,
                int(action),
                det_seed,
                playout_seed,
                checkpoint,
                roles,
                continuation,
            )
            rewards_by_action[int(action)].append(reward)
            wins_by_action[int(action)].append(win)

    rows = []
    score_matrix = []
    for action in actions:
        action = int(action)
        rewards = np.asarray(rewards_by_action[action], dtype=np.float32)
        wins = np.asarray(wins_by_action[action], dtype=np.float32)
        scores = wins + reward_score_weight * rewards
        score_matrix.append(scores)
        rows.append(
            {
                "action": action,
                "decoded": decode_action(action),
                "winrate": float(wins.mean()) if len(wins) else 0.0,
                "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
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


def decode_planned_event(env, agent: str, action: int) -> dict:
    if env._chancellor_pending and env.agent_selection == agent:
        pool = list(env._chancellor_pool)
        action_idx = int(action) - 900
        if len(pool) >= 3:
            keep_idx = action_idx // 2
            order_idx = action_idx % 2
        elif len(pool) == 2:
            keep_idx = action_idx
            order_idx = 0
        else:
            keep_idx = 0
            order_idx = 0
        if keep_idx >= len(pool) or keep_idx < 0:
            keep_idx = 0
        kept = pool[keep_idx] if pool else None
        returned = list(pool)
        if pool:
            returned.pop(keep_idx)
        if order_idx == 1:
            returned.reverse()
        return {
            "kind": "chancellor_choice",
            "kept_card": kept,
            "returned_cards": returned,
            "pool": pool,
        }

    hand = list(env._hands.get(agent, []))
    card = int(action) // 100
    if card not in hand:
        if 8 in hand and (5 in hand or 7 in hand):
            card = 8
        elif hand:
            card = hand[0]
    target_idx = (int(action) % 100) // 10
    guess = int(action) % 10
    target = f"player_{target_idx}" if target_idx < env.num_players else None
    if target and (target not in env.agents or env.terminations.get(target, True)):
        target = None
    remaining_hand = list(hand)
    if card in remaining_hand:
        remaining_hand.remove(card)
    return {
        "kind": "card",
        "card": int(card),
        "target": target,
        "target_idx": int(target_idx),
        "guess": int(guess),
        "hand": hand,
        "remaining_hand": remaining_hand,
        "target_card": int(env._hands[target][0]) if target and env._hands.get(target) else None,
        "known_target_card": known_card_for(env, agent, target) if target else None,
        "forced_countess": bool(card == 8 and (5 in hand or 7 in hand)),
    }


class TacticalTracker:
    def __init__(self, eval_agent: str):
        self.eval_agent = eval_agent
        self.counts = Counter()
        self.game_flags = Counter()
        self.priest_known = {}

    def before_eval_action(self, env, event: dict) -> None:
        if event["kind"] == "chancellor_choice":
            self.counts["chancellor_choices"] += 1
            self.game_flags["chancellor_used"] = 1
            kept = event.get("kept_card")
            returned = event.get("returned_cards", [])
            pool = event.get("pool", [])
            if kept is not None and pool and kept == max(pool):
                self.counts["chancellor_kept_highest"] += 1
            if kept == 9:
                self.counts["chancellor_kept_princess"] += 1
            if any(card >= 7 for card in returned):
                self.counts["chancellor_bottomed_king_plus"] += 1
            return

        card = event["card"]
        target = event.get("target")
        guess = event.get("guess")
        target_card = event.get("target_card")
        known = event.get("known_target_card")
        if card == 0:
            self.counts["spy_plays"] += 1
            self.game_flags["spy_played"] = 1
        elif card == 1 and target:
            self.counts["guard_attempts"] += 1
            if target_card == guess:
                self.counts["guard_hits"] += 1
            else:
                self.counts["guard_misses"] += 1
            if known is not None:
                self.counts["known_guard_attempts"] += 1
                if known == guess:
                    self.counts["known_guard_correct_guess"] += 1
                if target_card == guess:
                    self.counts["known_guard_hits"] += 1
            priest_card = self.priest_known.get(target)
            if priest_card is not None and known == priest_card and guess == priest_card:
                self.counts["priest_to_guard_attempts"] += 1
                if target_card == guess:
                    self.counts["priest_to_guard_hits"] += 1
        elif card == 2 and target:
            self.counts["priest_plays"] += 1
            if known is None:
                self.counts["priest_on_unknown"] += 1
            if target_card is not None:
                self.priest_known[target] = int(target_card)
        elif card == 3 and target:
            self.counts["baron_attempts"] += 1
            my_after = event.get("remaining_hand", [])
            my_val = my_after[0] if my_after else None
            if my_val is not None and target_card is not None:
                if my_val > target_card:
                    self.counts["baron_winning_comparisons"] += 1
                elif target_card > my_val:
                    self.counts["baron_losing_comparisons"] += 1
                else:
                    self.counts["baron_tie_comparisons"] += 1
        elif card == 6:
            self.counts["chancellor_plays"] += 1
            self.game_flags["chancellor_used"] = 1
        elif card == 8:
            if event.get("forced_countess"):
                self.counts["forced_countess_discards"] += 1
            else:
                self.counts["voluntary_countess_discards"] += 1
                self.game_flags["voluntary_countess"] = 1

    def observe_known_draw(self, pre_hand: list[int], post_hand: list[int], known_top: int | None) -> None:
        if known_top is None:
            return
        if post_hand != pre_hand and known_top in post_hand:
            self.counts["chancellor_known_draws"] += 1
            self.game_flags["chancellor_known_draw"] = 1

    def finish_game(self, env, won: bool) -> Counter:
        if self.eval_agent in getattr(env, "_round_winners", []):
            self.counts["main_round_wins"] += 1
        if getattr(env, "_round_spy_winner", None) == self.eval_agent:
            self.counts["spy_bonus_wins"] += 1
            self.game_flags["spy_bonus"] = 1
        if won:
            for flag in [
                "chancellor_used",
                "chancellor_known_draw",
                "spy_played",
                "spy_bonus",
                "voluntary_countess",
            ]:
                if self.game_flags.get(flag):
                    self.counts[f"{flag}_win_games"] += 1
        return self.counts


def summarize_tactical(counts: Counter) -> dict:
    def rate(num, den):
        return float(counts[num] / counts[den]) if counts[den] else 0.0

    return {
        "raw_counts": {key: int(value) for key, value in sorted(counts.items())},
        "guard_hit_rate": rate("guard_hits", "guard_attempts"),
        "known_guard_hit_rate": rate("known_guard_hits", "known_guard_attempts"),
        "known_guard_correct_guess_rate": rate("known_guard_correct_guess", "known_guard_attempts"),
        "priest_to_guard_hit_rate": rate("priest_to_guard_hits", "priest_to_guard_attempts"),
        "priest_on_unknown_rate": rate("priest_on_unknown", "priest_plays"),
        "baron_win_rate": rate("baron_winning_comparisons", "baron_attempts"),
        "baron_loss_rate": rate("baron_losing_comparisons", "baron_attempts"),
        "baron_tie_rate": rate("baron_tie_comparisons", "baron_attempts"),
        "chancellor_keep_highest_rate": rate("chancellor_kept_highest", "chancellor_choices"),
        "chancellor_known_draw_win_rate": rate("chancellor_known_draw_win_games", "chancellor_known_draws"),
        "spy_bonus_per_spy_play": rate("spy_bonus_wins", "spy_plays"),
        "voluntary_countess_win_rate": rate("voluntary_countess_win_games", "voluntary_countess_discards"),
    }


def summarize_outcomes(records: list[dict]) -> dict:
    rewards = [record["reward"] for record in records]
    wins = [record["won"] for record in records]
    summary = summarize_rewards(rewards, wins)
    outcome_counts = Counter(record["outcome"] for record in records)
    summary["outcomes"] = {
        key: {
            "count": int(outcome_counts[key]),
            "rate": float(outcome_counts[key] / max(1, len(records))),
        }
        for key in ["winner", "first_out", "second_out", "third_out", "final_loser", "unknown_loss"]
    }
    by_seat = {}
    for seat in [f"player_{i}" for i in range(4)]:
        seat_records = [record for record in records if record["seat"] == seat]
        if seat_records:
            by_seat[seat] = summarize_rewards(
                [record["reward"] for record in seat_records],
                [record["won"] for record in seat_records],
            )
    summary["by_seat"] = by_seat
    return summary


def classify_outcome(eval_agent: str, reward: float, elimination_order: list[str]) -> str:
    if reward >= 1.0:
        return "winner"
    if eval_agent in elimination_order:
        rank = elimination_order.index(eval_agent)
        return ["first_out", "second_out", "third_out"][min(rank, 2)]
    return "final_loser"


def prepare_policy_context(policy_name: str, args) -> dict:
    if policy_name == "heuristic_fair":
        return {}
    if policy_name == "step2_retarget":
        return {"checkpoint": resolve_checkpoint(args.step2_checkpoint)}
    if policy_name in {"step3_fast_dagger", "step3_hybrid_verify16"}:
        checkpoint_name = (
            args.step3_fast_checkpoint
            if policy_name == "step3_fast_dagger"
            else args.step3_hybrid_checkpoint
        )
        checkpoint, base_checkpoint, head, ckpt = load_advantage_bundle(checkpoint_name, None)
        verify_rollouts = 0 if policy_name == "step3_fast_dagger" else args.verify_rollouts
        return {
            "checkpoint": checkpoint,
            "base_checkpoint": base_checkpoint,
            "head": head,
            "ckpt": ckpt,
            "categories": ckpt.get("categories", []),
            "verify_rollouts": verify_rollouts,
            "max_actions": args.max_actions or int(ckpt.get("max_actions", 14)),
        }
    raise ValueError(policy_name)


def make_policy(policy_name: str, args, roles: dict[str, str], eval_agent: str, context: dict):
    if policy_name == "heuristic_fair":
        return FairHeuristicSeat()
    if policy_name == "step2_retarget":
        return RelativeModelSeat(context["checkpoint"])
    if policy_name in {"step3_fast_dagger", "step3_hybrid_verify16"}:
        policy_args = SimpleNamespace(**vars(args))
        policy_args.verify_rollouts = context["verify_rollouts"]
        policy_args.max_actions = context["max_actions"]
        return GenericAdvantageSeat(
            context["base_checkpoint"],
            context["head"],
            context["categories"],
            policy_args,
            roles,
            eval_agent,
        )
    raise ValueError(policy_name)


def evaluate_policy_config(policy_name: str, config_name: str, games: int, seed_start: int, args, context: dict) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_tactical = Counter()
    aggregate_advantage = Counter()
    aggregate_categories = defaultdict(Counter)

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policies = {eval_agent: make_policy(policy_name, args, roles, eval_agent, context)}
        tracker = TacticalTracker(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            action = role_action(env, agent, obs_dict, roles, policies, bot)
            event = decode_planned_event(env, agent, action)
            pre_eval_hand = list(env._hands.get(eval_agent, []))
            known_top = env._deck_knowledge.get(eval_agent, {}).get(0)

            if agent == eval_agent:
                tracker.before_eval_action(env, event)

            # Detect direct eliminations from the public pre-action state.
            direct_eliminated = []
            if event["kind"] == "card":
                card = event["card"]
                target = event.get("target")
                target_card = event.get("target_card")
                if card == 1 and target and target_card == event.get("guess"):
                    direct_eliminated.append(target)
                elif card == 3 and target:
                    my_after = event.get("remaining_hand", [])
                    my_val = my_after[0] if my_after else None
                    if my_val is not None and target_card is not None:
                        if my_val > target_card:
                            direct_eliminated.append(target)
                        elif target_card > my_val:
                            direct_eliminated.append(agent)
                elif card == 5 and target and target_card == 9:
                    direct_eliminated.append(target)
                elif card == 9:
                    direct_eliminated.append(agent)

            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        policy_obj = policies[eval_agent]
        if isinstance(policy_obj, GenericAdvantageSeat):
            aggregate_advantage.update(policy_obj.stats)
            for category, stats in policy_obj.category_stats.items():
                aggregate_categories[category].update(stats)
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "elimination_order": elimination_order,
                "roles": roles,
            }
        )

    summary = summarize_outcomes(records)
    summary["tactical"] = summarize_tactical(aggregate_tactical)
    if aggregate_advantage:
        summary["advantage"] = summarize_advantage(aggregate_advantage, aggregate_categories)
    return summary


def evaluate_policy(policy_name: str, args, logger: ExperimentLogger) -> dict:
    context = prepare_policy_context(policy_name, args)
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Seat-rotated fair arena avec metriques tactiques.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, config_name, args.games, seed_start, args, context)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque etape terminee.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"reward={result['mean_reward']:.4f}, "
                f"winner={result['outcomes']['winner']['rate']:.4f}"
            ),
            details={
                "arena": {
                    "winrate": result["winrate"],
                    "mean_reward": result["mean_reward"],
                    "outcomes": result["outcomes"],
                    "by_seat": result["by_seat"],
                },
                "tactical": result["tactical"],
            },
        )
    return {
        "configs": configs,
        "composite": composite_score(configs),
    }


def tactical_totals(policy: dict) -> dict:
    total = Counter()
    for config in policy["configs"].values():
        total.update(config["tactical"]["raw_counts"])
    return summarize_tactical(total)


def aggregate_outcomes(policy: dict) -> dict:
    total_games = sum(config["games"] for config in policy["configs"].values())
    counts = Counter()
    for config in policy["configs"].values():
        for outcome, row in config["outcomes"].items():
            counts[outcome] += row["count"]
    return {key: float(counts[key] / max(1, total_games)) for key in counts}


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Seat-Rotated Fair Tactical Arena",
        "",
        f"Date: {payload['created_at']}.",
        "",
        (
            "Le joueur evalue tourne entre `player_0`, `player_1`, `player_2`, "
            "`player_3`. Les heuristiques adverses utilisent "
            "`HeuristicBot(shuffle_targets=True)`."
        ),
        "",
        "Definitions rapides:",
        "",
        "- `Gagnant` utilise la meme definition que nos arenas historiques: reward final `>= 1.0`.",
        "- `Finaliste perdant` signifie que le joueur n'a pas ete elimine par effet de carte, mais perd a la resolution finale.",
        "- Les metriques tactiques ne mesurent que les decisions du joueur evalue, pas celles des adversaires.",
        "",
        "## Winrates",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, policy in payload["policies"].items():
        c = policy["configs"]
        lines.append(
            "| {name} | {a} | {b} | {c2} | {d} | {comp:.5f} |".format(
                name=POLICY_LABELS[name],
                a=pct(c["vs_0H_3R"]["winrate"]),
                b=pct(c["vs_1H_2R"]["winrate"]),
                c2=pct(c["vs_2H_1R"]["winrate"]),
                d=pct(c["vs_3H"]["winrate"]),
                comp=policy["composite"],
            )
        )

    lines.extend(
        [
            "",
            "## Winrates Par Siege En Full Heuristique",
            "",
            "| Politique | player_0 | player_1 | player_2 | player_3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        seats = policy["configs"]["vs_3H"]["by_seat"]
        lines.append(
            "| {name} | {p0} | {p1} | {p2} | {p3} |".format(
                name=POLICY_LABELS[name],
                p0=pct(seats["player_0"]["winrate"]),
                p1=pct(seats["player_1"]["winrate"]),
                p2=pct(seats["player_2"]["winrate"]),
                p3=pct(seats["player_3"]["winrate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Sorties Moyennes",
            "",
            "| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        outcomes = aggregate_outcomes(policy)
        lines.append(
            "| {name} | {winner} | {first} | {second} | {third} | {final} |".format(
                name=POLICY_LABELS[name],
                winner=pct(outcomes.get("winner", 0.0)),
                first=pct(outcomes.get("first_out", 0.0)),
                second=pct(outcomes.get("second_out", 0.0)),
                third=pct(outcomes.get("third_out", 0.0)),
                final=pct(outcomes.get("final_loser", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Metriques Tactiques Globales",
            "",
            "| Politique | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier pioche connue gagne | Espionne bonus / Espionne |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        tact = tactical_totals(policy)
        lines.append(
            "| {name} | {guard} | {known_guard} | {priest_guard} | {baron_win} | {baron_loss} | {chancellor_win} | {spy_bonus} |".format(
                name=POLICY_LABELS[name],
                guard=pct(tact["guard_hit_rate"]),
                known_guard=pct(tact["known_guard_hit_rate"]),
                priest_guard=pct(tact["priest_to_guard_hit_rate"]),
                baron_win=pct(tact["baron_win_rate"]),
                baron_loss=pct(tact["baron_loss_rate"]),
                chancellor_win=pct(tact["chancellor_known_draw_win_rate"]),
                spy_bonus=pct(tact["spy_bonus_per_spy_play"]),
            )
        )
    lines.extend(
        [
            "",
            "Definitions tactiques:",
            "",
            "- `Garde juste`: devinette exacte sur tous les Gardes joues.",
            "- `Garde connu juste`: Garde exact quand la carte de la cible etait connue dans l'etat du joueur.",
            "- `Pretre->Garde juste`: Garde exact sur une information obtenue precedemment par Pretre.",
            "- `Baron gagne/perdu`: comparaison favorable/defavorable au moment de jouer Baron.",
            "- `Chancelier pioche connue gagne`: proportion des pioches connues via Chancelier qui finissent dans une partie gagnee.",
            "- `Espionne bonus / Espionne`: bonus Espionne obtenu par Espionne jouee.",
        ]
    )

    lines.extend(
        [
            "",
            "## Note Interne - Comtesse Volontaire",
            "",
            "| Politique | Comtesses volontaires | Winrate apres Comtesse volontaire |",
            "|---|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        tact = tactical_totals(policy)
        raw = tact["raw_counts"]
        lines.append(
            "| {name} | {count} | {rate} |".format(
                name=POLICY_LABELS[name],
                count=raw.get("voluntary_countess_discards", 0),
                rate=pct(tact["voluntary_countess_win_rate"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Seat-rotated fair arena with tactical diagnostics.")
    parser.add_argument("--games", type=int, default=1000, help="Games per policy and composition, seats rotate inside.")
    parser.add_argument("--seed-start", type=int, default=250000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=16)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policies", nargs="+", default=list(POLICY_LABELS))
    parser.add_argument("--output", default="rotating_tactical_arena_1000.json")
    parser.add_argument("--markdown", default="rotating_tactical_arena_1000.md")
    parser.add_argument("--run-log", default="interlude_heuristic_comparison/logs/2026-04-25_rotating_tactical_arena_1000.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut rotating tactical arena",
        expected="Evaluer les modeles sans biais de siege et avec diagnostics tactiques.",
        actual=f"games={args.games}, policies={args.policies}",
        details=vars(args),
    )

    policies = {}
    for policy_name in args.policies:
        policies[policy_name] = evaluate_policy(policy_name, args, logger)
        logger.write(
            f"Modele termine - {policy_name}",
            expected="Un modele complet a passe les quatre compositions.",
            actual=f"composite={policies[policy_name]['composite']:.5f}",
        )

    payload = {
        "created_at": now_stamp(),
        "games_per_config": args.games,
        "seed_start": args.seed_start,
        "seat_rotation": "eval_agent = player_{game % 4}",
        "heuristic_mode": "shuffle_targets=True",
        "policies": policies,
    }
    payload = json_safe(payload)
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin rotating tactical arena",
        expected="Rapport JSON + Markdown pour analyse.",
        actual=f"json={output}, markdown={markdown}",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
