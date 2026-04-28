"""Evaluate Love Letter checkpoints against fixed opponent mixes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path


DEFAULT_CANDIDATES = {
    "current_champion": checkpoint_path("curriculum_phase1.pth"),
    "belief_bc": checkpoint_path("belief_conditioned_bc.pth"),
    "belief_ppo": checkpoint_path("belief_conditioned_ppo_final.pth"),
}

OPPONENT_CONFIGS = {
    "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
    "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
    "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
    "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
}


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def evaluate_config(main_actor, opponents, n_games=1000, seed_start=0, verbose=False):
    """Evaluate player_0 policy against a fixed 3-opponent table."""
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()

    rewards = np.zeros(n_games, dtype=np.float32)
    wins = np.zeros(n_games, dtype=np.int32)
    lengths = np.zeros(n_games, dtype=np.int32)

    for game in range(n_games):
        env.reset(seed=seed_start + game)
        main_state = None
        opponent_states = {f"player_{i}": None for i in range(1, 4)}
        n_actions = 0

        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()

            if agent == "player_0":
                rewards[game] += float(reward)

            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                action, main_state = main_actor.act(obs_dict, main_state, agent_id=agent)
                n_actions += 1
            else:
                opponent = opponents[agent]
                if opponent == "random":
                    action = random_action(obs_dict)
                elif opponent == "heuristic":
                    action = bot.choose_action(env, agent)
                elif hasattr(opponent, "act"):
                    action, opponent_states[agent] = opponent.act(
                        obs_dict,
                        opponent_states[agent],
                        agent_id=agent,
                    )
                else:
                    raise ValueError(f"Unknown opponent type: {opponent}")

            env.step(action)

        lengths[game] = n_actions
        wins[game] = int(rewards[game] >= 1.0)

        if verbose and (game + 1) % 500 == 0:
            print(f"    {game + 1}/{n_games} games")

    mean_reward = float(rewards.mean())
    reward_std = float(rewards.std())
    winrate = float(wins.mean())
    return {
        "games": n_games,
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(1.96 * np.sqrt(winrate * (1.0 - winrate) / n_games)),
        "mean_reward": mean_reward,
        "reward_std": reward_std,
        "reward_ci95": float(1.96 * reward_std / np.sqrt(n_games)),
        "avg_actions_player0": float(lengths.mean()),
    }


def candidate_paths(single_checkpoint=None):
    if single_checkpoint:
        path = Path(single_checkpoint)
        return {path.stem: path}
    return DEFAULT_CANDIDATES


def evaluate_models(candidates, games, seed_start, verbose=False):
    report = {}
    for name, path in candidates.items():
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue

        print(f"[load] {name}: {path}")
        actor = load_belief_policy(path)
        report[name] = {"path": str(path), "configs": {}}

        for config_name, opponents in OPPONENT_CONFIGS.items():
            result = evaluate_config(
                actor,
                opponents,
                n_games=games,
                seed_start=seed_start,
                verbose=verbose,
            )
            report[name]["configs"][config_name] = result
            print(
                f"  {config_name:10s} "
                f"winrate={result['winrate']:.3f} +/- {result['winrate_ci95']:.3f} "
                f"reward={result['mean_reward']:.3f} +/- {result['reward_ci95']:.3f}"
            )

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Love Letter checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Evaluate only one checkpoint.")
    parser.add_argument("--games", type=int, default=1000, help="Games per opponent config.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = {
        "games_per_config": args.games,
        "seed_start": args.seed_start,
        "models": evaluate_models(
            candidate_paths(args.checkpoint),
            games=args.games,
            seed_start=args.seed_start,
            verbose=args.verbose,
        ),
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
