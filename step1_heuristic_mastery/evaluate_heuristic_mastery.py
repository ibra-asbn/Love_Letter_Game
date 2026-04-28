"""Evaluate a heuristic student against randoms and HeuristicBot."""

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

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path
from step1_heuristic_mastery.common import (
    ExperimentLogger,
    STEP_CHECKPOINT_DIR,
    STEP_REPORT_DIR,
    absolute_to_relative_mask,
    composite_score,
    ensure_step_dirs,
    relative_to_absolute_action,
    now_stamp,
    resolve_step_path,
)


OPPONENT_CONFIGS = {
    "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
    "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
    "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
    "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
}


def resolve_checkpoint(name_or_path):
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = STEP_CHECKPOINT_DIR / name_or_path
    if candidate.exists():
        return candidate
    candidate = checkpoint_path(name_or_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


class StudentSeat:
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


def summarize_rewards(rewards, wins):
    rewards = np.asarray(rewards, dtype=np.float32)
    wins = np.asarray(wins, dtype=np.int32)
    winrate = float(wins.mean()) if len(wins) else 0.0
    return {
        "games": int(len(wins)),
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(1.96 * np.sqrt(winrate * (1.0 - winrate) / max(1, len(wins)))),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "reward_ci95": float(1.96 * rewards.std() / np.sqrt(max(1, len(rewards)))) if len(rewards) else 0.0,
    }


def evaluate_player0_student(checkpoint, opponents, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []
    lengths = []

    for game in range(games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        student = StudentSeat(checkpoint)
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
                action = student.act(obs_dict, agent)
                actions0 += 1
            else:
                opponent = opponents[agent]
                if opponent == "random":
                    action = random_action(obs_dict)
                elif opponent == "heuristic":
                    action = bot.choose_action(env, agent)
                else:
                    raise ValueError(opponent)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
    summary = summarize_rewards(rewards, wins)
    summary["avg_actions_player0"] = float(np.mean(lengths)) if lengths else 0.0
    return summary


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
            if agent == "player_0":
                action = bot.choose_action(env, agent)
            else:
                opponent = opponents[agent]
                action = bot.choose_action(env, agent) if opponent == "heuristic" else random_action(obs_dict)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
    return summarize_rewards(rewards, wins)


def evaluate_role_table(checkpoint, roles, games, seed_start):
    """Evaluate a table with roles: student, heuristic, random."""
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    role_rewards = defaultdict(list)
    role_wins = defaultdict(list)

    for game in range(games):
        np.random.seed(seed_start + game)
        env.reset(seed=seed_start + game)
        students = {
            agent: StudentSeat(checkpoint)
            for agent, role in roles.items()
            if role == "student"
        }
        rewards = {agent: 0.0 for agent in env.possible_agents}
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            role = roles[agent]
            if role == "student":
                action = students[agent].act(obs_dict, agent)
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


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Evaluate heuristic mastery.")
    parser.add_argument("--checkpoint", default="heuristic_student_attempt1.pth")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--output", default="heuristic_student_attempt1_eval.json")
    parser.add_argument("--run-log", default="step1_heuristic_mastery/logs/2026-04-24_step1_eval_attempt1.md")
    parser.add_argument("--skip-mirrors", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = resolve_step_path(args.output, STEP_REPORT_DIR)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut evaluation step1",
        expected=(
            "Comparer l'etudiant a HeuristicBot sur les memes tables, puis tester les miroirs "
            "student/heuristic."
        ),
        actual=f"checkpoint={checkpoint}, games={args.games}",
        details=vars(args),
    )

    student_configs = {}
    heuristic_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        student_configs[name] = evaluate_player0_student(checkpoint, opponents, args.games, args.seed_start)
        heuristic_configs[name] = evaluate_player0_heuristic(opponents, args.games, args.seed_start)
        logger.write(
            f"Config {name}",
            expected="L'etudiant doit se rapprocher de l'heuristique puis la depasser a terme.",
            actual=(
                f"student={student_configs[name]['winrate']:.3f}, "
                f"heuristic={heuristic_configs[name]['winrate']:.3f}"
            ),
            details={"student": student_configs[name], "heuristic": heuristic_configs[name]},
        )

    mirror_tables = {
        "heuristic_vs_3_students": {
            "player_0": "heuristic",
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
        "all_students": {
            "player_0": "student",
            "player_1": "student",
            "player_2": "student",
            "player_3": "student",
        },
        "all_heuristics": {
            "player_0": "heuristic",
            "player_1": "heuristic",
            "player_2": "heuristic",
            "player_3": "heuristic",
        },
    }
    mirror_results = {}
    if not args.skip_mirrors:
        mirror_results = {
            name: evaluate_role_table(checkpoint, roles, args.games, args.seed_start + 50_000)
            for name, roles in mirror_tables.items()
        }

    student_score = composite_score(student_configs)
    heuristic_score = composite_score(heuristic_configs)
    report = {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "games": args.games,
        "seed_start": args.seed_start,
        "student_configs": student_configs,
        "heuristic_configs": heuristic_configs,
        "student_composite": student_score,
        "heuristic_composite": heuristic_score,
        "student_minus_heuristic_composite": student_score - heuristic_score,
        "mirror_results": mirror_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation step1",
        expected="Avoir une lecture claire: etudiant >= heuristique ou non.",
        actual=(
            f"student_score={student_score:.4f}, heuristic_score={heuristic_score:.4f}, "
            f"delta={student_score - heuristic_score:.4f}"
        ),
        details=report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
