"""Collect sequence data labelled by HeuristicBot.

Unlike the old flat dataset, this keeps per-player decision sequences so the
student can train the recurrent hidden state it will actually use at inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import (
    ExperimentLogger,
    STEP_DATA_DIR,
    absolute_to_relative_action,
    absolute_to_relative_mask,
    ensure_step_dirs,
    now_stamp,
    resolve_step_path,
)


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def parse_mix(value):
    if value == "all":
        return [0, 1, 2, 3, 4]
    return [int(part) for part in value.split(",") if part.strip()]


def seat_uses_teacher(seat_idx, n_teacher_seats):
    return seat_idx < n_teacher_seats


def collect(args, logger):
    bot = HeuristicBot()
    env = LoveLetterRLEnv(num_players=4)
    mix = parse_mix(args.teacher_seat_mix)
    rng = np.random.default_rng(args.seed)

    sequences = []
    game_summaries = []
    transition_count = 0
    teacher_seat_counter = Counter()
    behavior_counter = Counter()

    for game in range(args.games):
        seed = args.seed + game
        env.reset(seed=seed)
        n_teacher_seats = int(rng.choice(mix))
        teacher_seat_counter[n_teacher_seats] += 1
        per_agent = {agent: [] for agent in env.possible_agents}
        total_rewards = {agent: 0.0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, info = env.last()
            total_rewards[agent] = total_rewards.get(agent, 0.0) + float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            teacher_action = bot.choose_action(env, agent)
            seat_idx = int(agent.rsplit("_", 1)[1])
            relative_teacher_action = absolute_to_relative_action(teacher_action, seat_idx)
            relative_mask = absolute_to_relative_mask(obs_dict["action_mask"], seat_idx)
            use_teacher = seat_uses_teacher(seat_idx, n_teacher_seats)
            behavior_action = teacher_action if use_teacher else random_action(obs_dict)
            behavior_counter["teacher" if use_teacher else "random"] += 1

            should_record = args.record_seats == "all" or (args.record_seats == "player0" and agent == "player_0")
            if should_record:
                per_agent[agent].append(
                    {
                        "obs": obs_dict["observation"].copy(),
                        "mask": relative_mask.copy(),
                        "action": int(relative_teacher_action),
                        "absolute_action": int(teacher_action),
                        "hidden_cards": info.get("hidden_cards", np.full(3, -1, dtype=np.int64)).copy(),
                        "behavior_action": int(behavior_action),
                        "behavior": "teacher" if use_teacher else "random",
                        "game": game,
                        "seat": seat_idx,
                    }
                )
                transition_count += 1
            env.step(behavior_action)

        for agent, items in per_agent.items():
            if not items:
                continue
            sequences.append(
                {
                    "game": game,
                    "agent": agent,
                    "seat": int(agent.rsplit("_", 1)[1]),
                    "teacher_seats_in_game": n_teacher_seats,
                    "steps": items,
                }
            )
        game_summaries.append(
            {
                "game": game,
                "seed": seed,
                "teacher_seats": n_teacher_seats,
                "rewards": total_rewards,
            }
        )

        if args.log_every_games and (game + 1) % args.log_every_games == 0:
            logger.write(
                "Collecte sequences heuristique",
                expected="Couvrir des etats generes par randoms, heuristiques et mixes.",
                actual=f"{game + 1}/{args.games} parties, {transition_count} transitions",
                details={
                    "sequences": len(sequences),
                    "teacher_seat_mix_seen": dict(teacher_seat_counter),
                    "behavior_counts": dict(behavior_counter),
                },
            )

    dataset = {
        "created_at": now_stamp(),
        "args": vars(args),
        "sequences": sequences,
        "game_summaries": game_summaries,
        "summary": {
            "games": args.games,
            "sequences": len(sequences),
            "transitions": transition_count,
            "teacher_seat_mix_seen": dict(teacher_seat_counter),
            "behavior_counts": dict(behavior_counter),
        },
    }
    return dataset


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Collect HeuristicBot sequence dataset.")
    parser.add_argument("--games", type=int, default=12000)
    parser.add_argument("--teacher-seat-mix", default="all", help="'all' or comma list from 0 to 4.")
    parser.add_argument("--record-seats", choices=["all", "player0"], default="all")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--output", default="teacher_sequences_attempt1.pkl")
    parser.add_argument("--report", default="teacher_sequences_attempt1_summary.json")
    parser.add_argument("--run-log", default="step1_heuristic_mastery/logs/2026-04-24_step1_collect_attempt1.md")
    parser.add_argument("--log-every-games", type=int, default=2000)
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut collecte step1",
        expected=(
            "Construire un dataset sequentiel de l'heuristique, avec split possible par partie "
            "et etats generes par plusieurs distributions d'adversaires."
        ),
        actual=f"games={args.games}, teacher_seat_mix={args.teacher_seat_mix}",
        details=vars(args),
    )
    dataset = collect(args, logger)

    output = resolve_step_path(args.output, STEP_DATA_DIR)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = resolve_step_path(args.report, STEP_DATA_DIR)
    report.write_text(json.dumps(dataset["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin collecte step1",
        expected="Sauvegarder un dataset riche et traçable.",
        actual=f"dataset={output}, report={report}",
        details=dataset["summary"],
    )
    print(json.dumps({"output": str(output), "report": str(report), **dataset["summary"]}, indent=2))


if __name__ == "__main__":
    main()
