"""Evaluate a rollout-guided step3 policy.

This is a deliberately direct action-value test: keep the strong step2 actor as
the default policy, then use small determinized rollouts on selected tactical
states to override only when the estimated action-value margin is clear.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import (
    ExperimentLogger,
    arena_summary,
    composite_score,
    now_stamp,
    resolve_checkpoint,
)
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS, ModelSeat, random_action, summarize_rewards
from step3_action_value.mini_rollout_probe import (
    choose_actions_for_probe,
    classify_state,
    decode_action,
    determinize_for_player,
)


STEP_DIR = PROJECT_ROOT / "step3_action_value"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def rollout_once(base_env, first_action, seed, checkpoint, opponents, player0_continuation):
    env = determinize_for_player(base_env, "player_0", seed)
    bot = HeuristicBot()
    model = ModelSeat(checkpoint) if player0_continuation == "model" else None
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
            if player0_continuation == "heuristic":
                action = bot.choose_action(env, agent)
            elif player0_continuation == "model":
                action = model.act(obs_dict, agent)
            elif player0_continuation == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(player0_continuation)
        else:
            action = opponent_step(env, agent, obs_dict, opponents, bot)
        env.step(action)

    return reward0, int(reward0 >= 1.0)


def evaluate_candidate_actions(env, actions, checkpoint, opponents, args, decision_seed):
    rows = []
    for action in actions:
        rewards = []
        wins = []
        for rollout in range(args.rollouts_per_action):
            seed = decision_seed * 1_000_000 + int(action) * 1000 + rollout
            reward, win = rollout_once(
                env,
                int(action),
                seed,
                checkpoint,
                opponents,
                args.player0_continuation,
            )
            rewards.append(reward)
            wins.append(win)
        rows.append(
            {
                "action": int(action),
                "decoded": decode_action(int(action)),
                "winrate": float(np.mean(wins)) if wins else 0.0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "reward_std": float(np.std(rewards)) if rewards else 0.0,
                "wins": int(sum(wins)),
                "rollouts": int(len(wins)),
            }
        )
    rows.sort(key=lambda row: (row["winrate"], row["mean_reward"]), reverse=True)
    return rows


class RolloutGuidedPlayer0:
    def __init__(self, checkpoint, opponents, args):
        self.checkpoint = checkpoint
        self.base = ModelSeat(checkpoint)
        self.bot = HeuristicBot()
        self.opponents = opponents
        self.args = args
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []

    def act(self, env, obs_dict, decision_seed):
        model_action = int(self.base.act(obs_dict, "player_0"))
        category = classify_state(env, "player_0")
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1

        if category not in set(self.args.categories):
            return model_action
        if int(obs_dict["action_mask"].sum()) <= 1:
            return model_action

        heuristic_action = int(self.bot.choose_action(env, "player_0"))
        candidates = choose_actions_for_probe(env, self.args.max_actions)
        for forced in [model_action, heuristic_action]:
            if forced not in candidates:
                candidates = [forced] + candidates
        candidates = list(dict.fromkeys(candidates))[: self.args.max_actions]

        rows = evaluate_candidate_actions(
            env,
            candidates,
            self.checkpoint,
            self.opponents,
            self.args,
            decision_seed,
        )
        by_action = {row["action"]: row for row in rows}
        model_row = by_action.get(model_action)
        if model_row is None:
            return model_action
        best = rows[0]
        margin = float(best["winrate"] - model_row["winrate"])
        reward_margin = float(best["mean_reward"] - model_row["mean_reward"])

        self.stats["guided_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += margin

        if margin >= self.args.override_margin and reward_margin >= self.args.min_reward_margin:
            chosen = int(best["action"])
            if chosen != model_action:
                self.stats["overrides"] += 1
                self.stats["sum_override_margin"] += margin
                self.category_stats[category]["overrides"] += 1
                self.category_stats[category]["sum_override_margin"] += margin
                if len(self.examples) < self.args.example_limit:
                    self.examples.append(
                        {
                            "category": category,
                            "model_action": model_action,
                            "model_decoded": decode_action(model_action),
                            "chosen_action": chosen,
                            "chosen_decoded": decode_action(chosen),
                            "margin": margin,
                            "reward_margin": reward_margin,
                            "top_actions": rows[:5],
                        }
                    )
            return chosen
        return model_action


def evaluate_player0_guided(checkpoint, opponents, games, seed_start, args):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []
    lengths = []
    aggregate_stats = Counter()
    aggregate_categories = defaultdict(Counter)
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        guided = RolloutGuidedPlayer0(checkpoint, opponents, args)
        reward0 = 0.0
        actions0 = 0
        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                action = guided.act(env, obs_dict, decision_seed=seed * 100 + turn)
                actions0 += 1
            else:
                action = opponent_step(env, agent, obs_dict, opponents, bot)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
        aggregate_stats.update(guided.stats)
        for category, stats in guided.category_stats.items():
            aggregate_categories[category].update(stats)
        if len(examples) < args.example_limit:
            examples.extend(guided.examples[: args.example_limit - len(examples)])

    summary = summarize_rewards(rewards, wins, lengths)
    summary["guidance"] = summarize_guidance(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def summarize_guidance(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["guided_checks"])
    overrides = max(1, stats["overrides"])
    return {
        "decisions": int(stats["decisions"]),
        "guided_checks": int(stats["guided_checks"]),
        "overrides": int(stats["overrides"]),
        "check_rate": float(stats["guided_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "mean_override_margin": float(stats["sum_override_margin"] / overrides) if stats["overrides"] else 0.0,
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "mean_check_margin": float(row["sum_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_margin": float(row["sum_override_margin"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


def run_guided_evaluation(checkpoint, games, seed_start, args, logger=None):
    configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        configs[name] = evaluate_player0_guided(checkpoint, opponents, games, seed_start, args)
        if logger:
            logger.write(
                f"Config {name}",
                expected="Le search tactique doit ameliorer le score sans trop d'overrides.",
                actual=(
                    f"guided={configs[name]['winrate']:.4f}, "
                    f"overrides={configs[name]['guidance']['overrides']}"
                ),
                details=configs[name],
            )
    return {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "games": games,
        "seed_start": seed_start,
        "args": vars(args),
        "model_configs": configs,
        "model_composite": composite_score(configs),
    }


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate rollout-guided step3 policy.")
    parser.add_argument("--checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=790000)
    parser.add_argument("--categories", nargs="+", default=["baron", "guard", "prince"])
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--override-margin", type=float, default=0.12)
    parser.add_argument("--min-reward-margin", type=float, default=-999.0)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--example-limit", type=int, default=25)
    parser.add_argument("--output", default="step3_rollout_guided_eval.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-24_step3_rollout_guided_eval.md")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation rollout-guided",
        expected=(
            "Tester si une action-value par rollouts peut battre le Step2 brut avant distillation."
        ),
        actual=f"checkpoint={checkpoint}, games={args.games}",
        details=vars(args),
    )

    report = run_guided_evaluation(checkpoint, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation rollout-guided",
        expected="Obtenir un verdict: signal action-value exploitable ou non.",
        actual=f"guided_score={report['model_composite']:.5f}",
        details={"model": arena_summary(report["model_configs"])},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
