"""Evaluate step 2 candidates against random and heuristic compositions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import absolute_to_relative_mask, relative_to_absolute_action
from step2_rl_finetune.common import (
    ExperimentLogger,
    STEP_REPORT_DIR,
    arena_summary,
    composite_score,
    ensure_step_dirs,
    now_stamp,
    resolve_checkpoint,
    resolve_step_path,
)


OPPONENT_CONFIGS = {
    "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
    "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
    "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
    "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
}


ROLE_TABLES = {
    "student_vs_3_students": {
        "player_0": "student",
        "player_1": "student",
        "player_2": "student",
        "player_3": "student",
    },
    "student_student_vs_heuristic_heuristic": {
        "player_0": "student",
        "player_1": "student",
        "player_2": "heuristic",
        "player_3": "heuristic",
    },
    "heuristic_vs_3_students": {
        "player_0": "heuristic",
        "player_1": "student",
        "player_2": "student",
        "player_3": "student",
    },
    "student_vs_3_heuristics": {
        "player_0": "student",
        "player_1": "heuristic",
        "player_2": "heuristic",
        "player_3": "heuristic",
    },
}


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


class ModelSeat:
    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None

    def act(self, obs_dict, agent):
        my_idx = int(agent.rsplit("_", 1)[1])
        relative_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        relative_action, self.state = self.policy.act(relative_obs, self.state, agent_id=agent)
        return relative_to_absolute_action(relative_action, my_idx)


def summarize_rewards(rewards, wins, lengths=None):
    rewards = np.asarray(rewards, dtype=np.float32)
    wins = np.asarray(wins, dtype=np.int32)
    winrate = float(wins.mean()) if len(wins) else 0.0
    summary = {
        "games": int(len(wins)),
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(1.96 * np.sqrt(winrate * (1.0 - winrate) / max(1, len(wins)))),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "reward_ci95": float(1.96 * rewards.std() / np.sqrt(max(1, len(rewards)))) if len(rewards) else 0.0,
    }
    if lengths is not None:
        summary["avg_actions_player0"] = float(np.mean(lengths)) if lengths else 0.0
    return summary


def evaluate_player0_model(checkpoint, opponents, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []
    lengths = []

    for game in range(games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        model = ModelSeat(checkpoint)
        reward0 = 0.0
        actions0 = 0
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                action = model.act(obs_dict, agent)
                actions0 += 1
            else:
                opponent = opponents[agent]
                if opponent == "heuristic":
                    action = bot.choose_action(env, agent)
                elif opponent == "random":
                    action = random_action(obs_dict)
                else:
                    raise ValueError(opponent)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
    return summarize_rewards(rewards, wins, lengths)


def evaluate_player0_heuristic(opponents, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []

    for game in range(games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        reward0 = 0.0
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0" or opponents[agent] == "heuristic":
                action = bot.choose_action(env, agent)
            else:
                action = random_action(obs_dict)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
    return summarize_rewards(rewards, wins)


def evaluate_role_table(checkpoint, roles, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    role_rewards = defaultdict(list)
    role_wins = defaultdict(list)

    for game in range(games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        models = {agent: ModelSeat(checkpoint) for agent, role in roles.items() if role == "student"}
        rewards = {agent: 0.0 for agent in env.possible_agents}
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            role = roles[agent]
            if role == "student":
                action = models[agent].act(obs_dict, agent)
            elif role == "heuristic":
                action = bot.choose_action(env, agent)
            elif role == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(role)
            env.step(action)
        for agent, role in roles.items():
            role_rewards[role].append(rewards[agent])
            role_wins[role].append(int(rewards[agent] >= 1.0))

    return {
        role: summarize_rewards(role_rewards[role], role_wins[role])
        for role in sorted(role_rewards)
    }


def run_evaluation(checkpoint, games, seed_start, include_role_tables=True, role_games=None, logger=None):
    model_configs = {}
    heuristic_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        model_configs[name] = evaluate_player0_model(checkpoint, opponents, games, seed_start)
        heuristic_configs[name] = evaluate_player0_heuristic(opponents, games, seed_start)
        if logger:
            logger.write(
                f"Config {name}",
                expected="Mesurer si le candidat depasse le teacher sur cette composition.",
                actual=(
                    f"model={model_configs[name]['winrate']:.4f}, "
                    f"heuristic={heuristic_configs[name]['winrate']:.4f}"
                ),
                details={"model": model_configs[name], "heuristic": heuristic_configs[name]},
            )

    role_results = {}
    if include_role_tables:
        role_games = games if role_games is None else role_games
        role_results = {
            name: evaluate_role_table(checkpoint, roles, role_games, seed_start + 50_000)
            for name, roles in ROLE_TABLES.items()
        }

    model_score = composite_score(model_configs)
    heuristic_score = composite_score(heuristic_configs)
    return {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "games": games,
        "seed_start": seed_start,
        "model_configs": model_configs,
        "heuristic_configs": heuristic_configs,
        "model_composite": model_score,
        "heuristic_composite": heuristic_score,
        "model_minus_heuristic_composite": model_score - heuristic_score,
        "role_results": role_results,
    }


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Evaluate a step2 candidate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--role-games", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=600000)
    parser.add_argument("--output", default="step2_eval.json")
    parser.add_argument("--run-log", default=None)
    parser.add_argument("--skip-role-tables", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = resolve_step_path(args.output, STEP_REPORT_DIR)
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation step2",
        expected="Figer le niveau du candidat contre randoms, mixes heuristiques, et tables de roles.",
        actual=f"checkpoint={checkpoint}, games={args.games}",
        details=vars(args),
    )

    report = run_evaluation(
        checkpoint=checkpoint,
        games=args.games,
        seed_start=args.seed_start,
        include_role_tables=not args.skip_role_tables,
        role_games=args.role_games,
        logger=logger,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation step2",
        expected="Avoir le score de depart et le delta vs HeuristicBot.",
        actual=(
            f"model_score={report['model_composite']:.5f}, "
            f"heuristic_score={report['heuristic_composite']:.5f}, "
            f"delta={report['model_minus_heuristic_composite']:.5f}"
        ),
        details={
            "model": arena_summary(report["model_configs"]),
            "heuristic": arena_summary(report["heuristic_configs"]),
            "role_results": report["role_results"],
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

