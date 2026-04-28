"""Counterfactual tests for using a checkpoint's own belief probabilities.

The goal is not to create a new champion. It asks a narrower question:
if the actor had used the belief head more cleanly at decision time, would
performance improve?
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path


PARIS_TZ = ZoneInfo("Europe/Paris")
CARD_NAMES = [
    "Espionne",
    "Garde",
    "Pretre",
    "Baron",
    "Servante",
    "Prince",
    "Chancelier",
    "Roi",
    "Comtesse",
    "Princesse",
]

OPPONENT_CONFIGS = {
    "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
    "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
    "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
    "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
}


def now_stamp():
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


class ExperimentLogger:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, title, expected=None, actual=None, details=None):
        message = f"[{now_stamp()}] {title}"
        print(message, flush=True)
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"## {message}\n\n")
            if expected is not None:
                f.write(f"**Attendu**: {expected}\n\n")
            if actual is not None:
                f.write(f"**Obtenu**: {actual}\n\n")
            if details is not None:
                f.write("```json\n")
                f.write(json.dumps(details, indent=2, ensure_ascii=False))
                f.write("\n```\n\n")


@dataclass
class PolicyStats:
    decisions: int = 0
    changed_actions: int = 0
    changes_by_reason: Counter = field(default_factory=Counter)
    raw_card_counts: Counter = field(default_factory=Counter)
    guided_card_counts: Counter = field(default_factory=Counter)
    guard_raw: int = 0
    guard_guided: int = 0
    guard_raw_top_belief: int = 0
    guard_guided_top_belief: int = 0
    guard_raw_hits: int = 0
    guard_guided_hits: int = 0

    def to_dict(self):
        return {
            "decisions": self.decisions,
            "changed_actions": self.changed_actions,
            "change_rate": self.changed_actions / self.decisions if self.decisions else 0.0,
            "changes_by_reason": dict(self.changes_by_reason),
            "raw_card_counts": dict(self.raw_card_counts),
            "guided_card_counts": dict(self.guided_card_counts),
            "guard": {
                "raw": self.guard_raw,
                "guided": self.guard_guided,
                "raw_top_belief": self.guard_raw_top_belief,
                "guided_top_belief": self.guard_guided_top_belief,
                "raw_hits": self.guard_raw_hits,
                "guided_hits": self.guard_guided_hits,
                "raw_top_belief_rate": self.guard_raw_top_belief / self.guard_raw if self.guard_raw else 0.0,
                "guided_top_belief_rate": self.guard_guided_top_belief / self.guard_guided if self.guard_guided else 0.0,
                "raw_hit_rate": self.guard_raw_hits / self.guard_raw if self.guard_raw else 0.0,
                "guided_hit_rate": self.guard_guided_hits / self.guard_guided if self.guard_guided else 0.0,
            },
        }


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def extract_belief(policy):
    debug = getattr(policy, "last_debug", None)
    if debug is None:
        return None
    if hasattr(debug, "belief_probs") and debug.belief_probs is not None:
        return debug.belief_probs.squeeze(0).numpy()
    if hasattr(debug, "probs") and debug.probs is not None:
        return debug.probs
    return None


def cards_from_obs(obs):
    cards = []
    for card, value in enumerate(obs[:10]):
        count = int(round(float(value) * 3.0))
        cards.extend([card] * max(0, count))
    return cards


def kept_card_after_play(hand, played_card):
    remaining = list(hand)
    if played_card in remaining:
        remaining.remove(played_card)
    return remaining[0] if remaining else None


def target_belief_dim(my_idx, target_idx):
    if target_idx >= 4 or target_idx == my_idx:
        return None
    rel = (target_idx - my_idx) % 4
    return rel - 1 if rel else None


def decode_card(action):
    if action >= 900:
        return "ChancellorChoice"
    card = action // 100
    return CARD_NAMES[card] if 0 <= card < len(CARD_NAMES) else str(card)


def best_guard_guess(probs):
    scores = np.array(probs, copy=True)
    scores[1] = -1.0
    return int(np.argmax(scores))


def chancellor_choice(pool, mask):
    if not pool:
        return None
    best_idx = max(range(len(pool)), key=lambda i: pool[i] if pool[i] != 9 else -0.5)
    action = 900 + best_idx * 2 if len(pool) >= 3 else 900 + best_idx
    return int(action) if mask[action] == 1 else None


def action_score(action, hand, belief, my_idx):
    if action >= 900 or belief is None:
        return -1e9

    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    dim = target_belief_dim(my_idx, target_idx)
    probs = belief[dim] if dim is not None else None
    kept = kept_card_after_play(hand, card)

    if card == 1 and probs is not None and guess != 1:
        top = best_guard_guess(probs)
        score = 4.0 * float(probs[guess])
        if guess == top:
            score += 1.0
        return score

    if card == 2 and probs is not None:
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)))) / np.log(10.0)
        return 0.25 * entropy

    if card == 3 and probs is not None and kept is not None:
        p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
        p_equal = float(probs[kept])
        p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
        return 3.0 * p_lower - 5.0 * p_higher - 0.5 * p_equal

    if card == 5:
        if target_idx == my_idx:
            if kept == 9:
                return -5.0
            if kept is not None and kept <= 2:
                return 0.8
            if kept is not None and kept <= 4:
                return 0.3
            return -0.4
        if probs is not None:
            return 5.0 * float(probs[9]) + 1.5 * float(probs[8]) + 0.5 * float(probs[7])

    if card == 7 and probs is not None and kept is not None:
        expected = float(np.sum(probs * np.arange(10, dtype=np.float32)))
        return 2.0 * ((expected - kept) / 9.0)

    return -1e9


def best_retarget_action(raw_action, mask, hand, belief, my_idx, env):
    if raw_action >= 900:
        action = chancellor_choice(list(env._chancellor_pool), mask)
        return action, "chancellor_visible_best" if action is not None and action != raw_action else None

    card = raw_action // 100
    if card not in {1, 3, 5, 7}:
        return None, None

    valid_same_card = [int(a) for a in np.where(mask == 1)[0] if int(a) < 900 and int(a) // 100 == card]
    if not valid_same_card:
        return None, None

    scored = [(action_score(a, hand, belief, my_idx), a) for a in valid_same_card]
    best_score, best_action = max(scored, key=lambda item: item[0])
    raw_score = action_score(raw_action, hand, belief, my_idx)
    if best_score > raw_score + 1e-6 and best_score > -1e8:
        return int(best_action), f"retarget_{CARD_NAMES[card]}"
    return None, None


def best_tactical_action(raw_action, mask, hand, belief, my_idx, threshold, margin):
    valid_actions = [int(a) for a in np.where(mask == 1)[0]]
    scored = [(action_score(a, hand, belief, my_idx), a) for a in valid_actions]
    best_score, best_action = max(scored, key=lambda item: item[0])
    raw_score = action_score(raw_action, hand, belief, my_idx)
    if best_score >= threshold and best_score > raw_score + margin:
        return int(best_action), f"tactical_{CARD_NAMES[best_action // 100]}"
    return None, None


class CounterfactualPolicy:
    def __init__(self, checkpoint, mode, threshold=1.0, margin=0.25):
        self.base = load_belief_policy(checkpoint)
        self.mode = mode
        self.threshold = threshold
        self.margin = margin
        self.stats = PolicyStats()

    def eval(self):
        self.base.eval()
        return self

    def act(self, obs_dict, hidden_state=None, agent_id="player_0", env=None):
        raw_action, new_state = self.base.act(obs_dict, hidden_state, agent_id=agent_id)
        belief = extract_belief(self.base)
        mask = obs_dict["action_mask"]
        hand = cards_from_obs(obs_dict["observation"])
        my_idx = int(agent_id.rsplit("_", 1)[1])

        guided_action = raw_action
        reason = None
        if self.mode == "retarget":
            guided_action, reason = best_retarget_action(raw_action, mask, hand, belief, my_idx, env)
            if guided_action is None:
                guided_action = raw_action
        elif self.mode == "tactical":
            guided_action, reason = best_tactical_action(
                raw_action,
                mask,
                hand,
                belief,
                my_idx,
                self.threshold,
                self.margin,
            )
            if guided_action is None:
                guided_action = raw_action
        elif self.mode != "raw":
            raise ValueError(f"Unknown mode: {self.mode}")

        self._record_decision(raw_action, guided_action, reason, belief, my_idx, env)
        return int(guided_action), new_state

    def _record_decision(self, raw_action, guided_action, reason, belief, my_idx, env):
        self.stats.decisions += 1
        self.stats.raw_card_counts[decode_card(raw_action)] += 1
        self.stats.guided_card_counts[decode_card(guided_action)] += 1
        if raw_action != guided_action:
            self.stats.changed_actions += 1
            self.stats.changes_by_reason[reason or "changed"] += 1

        self._record_guard(raw_action, belief, my_idx, env, raw=True)
        self._record_guard(guided_action, belief, my_idx, env, raw=False)

    def _record_guard(self, action, belief, my_idx, env, raw):
        if action >= 900 or action // 100 != 1 or belief is None:
            return
        target_idx = (action % 100) // 10
        guess = action % 10
        dim = target_belief_dim(my_idx, target_idx)
        if dim is None:
            return
        target = f"player_{target_idx}"
        target_hand = list(env._hands.get(target, [])) if env is not None else []
        target_card = target_hand[0] if target_hand else None
        top_guess = best_guard_guess(belief[dim])

        if raw:
            self.stats.guard_raw += 1
            if guess == top_guess:
                self.stats.guard_raw_top_belief += 1
            if target_card is not None and guess == target_card:
                self.stats.guard_raw_hits += 1
        else:
            self.stats.guard_guided += 1
            if guess == top_guess:
                self.stats.guard_guided_top_belief += 1
            if target_card is not None and guess == target_card:
                self.stats.guard_guided_hits += 1


def evaluate_policy(policy, opponents, n_games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = np.zeros(n_games, dtype=np.float32)
    wins = np.zeros(n_games, dtype=np.int32)
    lengths = np.zeros(n_games, dtype=np.int32)

    for game in range(n_games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        main_state = None
        n_actions = 0

        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                rewards[game] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                action, main_state = policy.act(obs_dict, main_state, agent_id=agent, env=env)
                n_actions += 1
            else:
                opponent = opponents[agent]
                if opponent == "random":
                    action = random_action(obs_dict)
                elif opponent == "heuristic":
                    action = bot.choose_action(env, agent)
                else:
                    raise ValueError(f"Unknown opponent type: {opponent}")

            env.step(action)

        lengths[game] = n_actions
        wins[game] = int(rewards[game] >= 1.0)

    winrate = float(wins.mean())
    reward_std = float(rewards.std())
    return {
        "games": int(n_games),
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(1.96 * np.sqrt(winrate * (1.0 - winrate) / n_games)),
        "mean_reward": float(rewards.mean()),
        "reward_std": reward_std,
        "reward_ci95": float(1.96 * reward_std / np.sqrt(n_games)),
        "avg_actions_player0": float(lengths.mean()),
    }


def composite_score(configs):
    weights = {"vs_0H_3R": 0.10, "vs_1H_2R": 0.20, "vs_2H_1R": 0.30, "vs_3H": 0.40}
    return float(sum(weights[name] * configs[name]["winrate"] for name in weights))


def evaluate_mode(checkpoint, mode, games, seed_start, threshold, margin, logger):
    policy = CounterfactualPolicy(checkpoint, mode=mode, threshold=threshold, margin=margin).eval()
    configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        result = evaluate_policy(policy, opponents, games, seed_start)
        configs[name] = result
        logger.write(
            f"Counterfactual {mode} - {name}",
            expected="Comparer la perf avec les memes probas de belief mais une decision plus coherente.",
            actual=f"winrate={result['winrate']:.3f}, reward={result['mean_reward']:.3f}",
            details={
                "wins": result["wins"],
                "games": result["games"],
                "winrate_ci95": result["winrate_ci95"],
                "reward_ci95": result["reward_ci95"],
            },
        )
    return {
        "mode": mode,
        "threshold": threshold if mode == "tactical" else None,
        "margin": margin if mode == "tactical" else None,
        "composite_score": composite_score(configs),
        "configs": configs,
        "policy_stats": policy.stats.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate belief-use counterfactual policies.")
    parser.add_argument("--checkpoint", default=str(checkpoint_path("champion_belief_ppo_attempt2_tactical_best.pth")))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--modes", nargs="+", default=["raw", "retarget", "tactical"])
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-log", default=None)
    args = parser.parse_args()

    checkpoint = str(checkpoint_path(args.checkpoint))
    logger = ExperimentLogger(args.run_log)
    logger.write(
        "Debut evaluation contre-factuelle belief",
        expected=(
            "Verifier si les probabilites du belief ameliorent les performances quand "
            "elles pilotent proprement les cibles/devinettes."
        ),
        actual=f"checkpoint={checkpoint}, games={args.games}, modes={args.modes}",
        details=vars(args),
    )

    report = {
        "checkpoint": checkpoint,
        "games_per_config": args.games,
        "seed_start": args.seed_start,
        "created_at": now_stamp(),
        "modes": {},
    }
    for mode in args.modes:
        report["modes"][mode] = evaluate_mode(
            checkpoint,
            mode,
            args.games,
            args.seed_start,
            args.threshold,
            args.margin,
            logger,
        )

    if "raw" in report["modes"]:
        raw_configs = report["modes"]["raw"]["configs"]
        for mode, mode_report in report["modes"].items():
            if mode == "raw":
                continue
            mode_report["delta_vs_raw"] = {
                name: {
                    "winrate": mode_report["configs"][name]["winrate"] - raw_configs[name]["winrate"],
                    "mean_reward": mode_report["configs"][name]["mean_reward"] - raw_configs[name]["mean_reward"],
                }
                for name in OPPONENT_CONFIGS
            }
            mode_report["composite_delta_vs_raw"] = (
                mode_report["composite_score"] - report["modes"]["raw"]["composite_score"]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin evaluation contre-factuelle belief",
        expected="Si le belief est utile et mal exploite, retarget/tactical doit battre raw.",
        actual="Rapport JSON ecrit.",
        details={
            mode: {
                "composite_score": values["composite_score"],
                "delta_vs_raw": values.get("composite_delta_vs_raw"),
                "changed_actions": values["policy_stats"]["changed_actions"],
                "change_rate": values["policy_stats"]["change_rate"],
            }
            for mode, values in report["modes"].items()
        },
    )
    print(json.dumps(report["modes"], indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
