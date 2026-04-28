"""Statistically stable baseline evaluations for Love Letter policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def play_game(env, main_policy, seed):
    env.reset(seed=seed)
    bot = HeuristicBot()
    rewards = {agent: 0.0 for agent in env.possible_agents}

    for agent in env.agent_iter():
        obs_dict, reward, terminated, truncated, _info = env.last()
        rewards[agent] += float(reward)

        if terminated or truncated:
            env.step(None)
            continue

        if agent == "player_0" and main_policy == "heuristic":
            action = bot.choose_action(env, agent)
        else:
            action = random_action(obs_dict)
        env.step(action)

    return rewards["player_0"]


def summarize(values):
    rewards = np.asarray(values, dtype=np.float32)
    wins = rewards >= 1.0
    n = len(rewards)
    winrate = float(wins.mean())
    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std())
    reward_ci95 = 1.96 * reward_std / np.sqrt(n)
    winrate_ci95 = 1.96 * np.sqrt(winrate * (1.0 - winrate) / n)
    return {
        "games": n,
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(winrate_ci95),
        "mean_reward": reward_mean,
        "reward_std": reward_std,
        "reward_ci95": float(reward_ci95),
    }


def evaluate(games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    results = {}
    for policy in ("random", "heuristic"):
        rewards = []
        for idx in range(games):
            rewards.append(play_game(env, policy, seed_start + idx))
            if games >= 10_000 and (idx + 1) % 5_000 == 0:
                print(f"{policy}: {idx + 1}/{games} games")
        results[f"{policy}_vs_3_random"] = summarize(rewards)

    random_result = results["random_vs_3_random"]
    heuristic_result = results["heuristic_vs_3_random"]
    results["lift"] = {
        "winrate_abs": heuristic_result["winrate"] - random_result["winrate"],
        "winrate_rel": heuristic_result["winrate"] / random_result["winrate"]
        if random_result["winrate"] > 0
        else None,
        "mean_reward_abs": heuristic_result["mean_reward"] - random_result["mean_reward"],
        "mean_reward_rel": heuristic_result["mean_reward"] / random_result["mean_reward"]
        if random_result["mean_reward"] > 0
        else None,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random and HeuristicBot vs 3 random players.")
    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate(args.games, args.seed_start)
    payload = {
        "date": "2026-04-24",
        "games_per_policy": args.games,
        "seed_start": args.seed_start,
        "results": results,
    }

    print(json.dumps(payload, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
