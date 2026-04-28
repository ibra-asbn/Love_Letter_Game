"""Asymmetric lineage matchups for the current champion.

Each matchup puts one singleton policy against three copies of another policy,
while rotating the singleton across seats. This answers questions like:

- Champion vs 3 Step3
- Step3 vs 3 Champions

It is evaluation-only; no model is updated here.
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

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import (
    TacticalTracker,
    classify_outcome,
    decode_planned_event,
    prepare_policy_context,
    summarize_tactical,
)
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step5_execution_heads.chancellor_head import load_chancellor_head
from step5_execution_heads.cards.baron.evaluate_baron_specialist import direct_eliminations_from_event
from step6_self_play.lineage_arena import POLICY_LABELS, make_lineage_policy, pct, policy_action


STEP_DIR = PROJECT_ROOT / "step6_self_play"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"

MATCHUPS = [
    ("champion_vs_3_step3", "champion_cbp", "step3_fast"),
    ("champion_vs_3_step2", "champion_cbp", "step2_retarget"),
    ("champion_vs_3_heuristic", "champion_cbp", "heuristic_fair"),
    ("champion_vs_3_curriculum", "champion_cbp", "curriculum_phase1"),
    ("step3_vs_3_champions", "step3_fast", "champion_cbp"),
    ("step2_vs_3_champions", "step2_retarget", "champion_cbp"),
]


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def ci95(winrate: float, n: int) -> float:
    return float(1.96 * np.sqrt(winrate * (1.0 - winrate) / max(1, n)))


def summarize_group(records: list[dict], tactical: Counter) -> dict:
    wins = sum(row["won"] for row in records)
    rewards = [row["reward"] for row in records]
    outcomes = Counter(row["outcome"] for row in records)
    by_seat = defaultdict(list)
    for row in records:
        by_seat[row["seat"]].append(row)
    winrate = float(wins / max(1, len(records)))
    tact = summarize_tactical(tactical)
    raw = tact["raw_counts"]
    games = max(1, len(records))
    return {
        "games": int(len(records)),
        "score_ge_1": winrate,
        "score_ge_1_ci95": ci95(winrate, len(records)),
        "main_round_win_rate": float(raw.get("main_round_wins", 0) / games),
        "spy_bonus_rate": float(raw.get("spy_bonus_wins", 0) / games),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "outcomes": {
            key: {
                "count": int(outcomes[key]),
                "rate": float(outcomes[key] / games),
            }
            for key in ["winner", "first_out", "second_out", "third_out", "final_loser"]
        },
        "by_seat": {
            seat: {
                "games": int(len(rows)),
                "score_ge_1": float(sum(row["won"] for row in rows) / max(1, len(rows))),
            }
            for seat, rows in sorted(by_seat.items())
        },
        "tactical": tact,
    }


def evaluate_matchup(name: str, singleton_policy: str, opponent_policy: str, args, contexts: dict, logger: ExperimentLogger) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    seats = [f"player_{idx}" for idx in range(4)]
    records = []
    tactical = {"singleton": Counter(), "opponents": Counter()}
    champion_head_stats = {
        "singleton_champion_chancellor": Counter(),
        "singleton_champion_baron": Counter(),
        "singleton_champion_prince": Counter(),
        "opponent_champion_chancellor": Counter(),
        "opponent_champion_baron": Counter(),
        "opponent_champion_prince": Counter(),
    }

    for game in range(args.games_per_matchup):
        seed = args.seed_start + args.matchup_seed_stride * MATCHUP_INDEX[name] + game
        np.random.seed(seed)
        singleton_seat = seats[game % len(seats)]
        starting_agent = seats[(game // len(seats)) % len(seats)]
        seat_to_policy = {
            seat: singleton_policy if seat == singleton_seat else opponent_policy
            for seat in seats
        }
        seat_to_group = {
            seat: "singleton" if seat == singleton_seat else "opponents"
            for seat in seats
        }
        env.reset(seed=seed, options={"starting_agent": starting_agent})
        policies = {
            seat: make_lineage_policy(policy_name, seat, args, contexts)
            for seat, policy_name in seat_to_policy.items()
        }
        trackers = {seat: TacticalTracker(seat) for seat in seats}
        rewards = {seat: 0.0 for seat in seats}
        elimination_order = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            policy_name = seat_to_policy[agent]
            action = policy_action(policy_name, policies[agent], env, obs_dict, agent)
            event = decode_planned_event(env, agent, action)
            pre_hands = {seat: list(env._hands.get(seat, [])) for seat in seats}
            known_tops = {seat: env._deck_knowledge.get(seat, {}).get(0) for seat in seats}
            trackers[agent].before_eval_action(env, event)
            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            for seat in seats:
                trackers[seat].observe_known_draw(pre_hands[seat], list(env._hands.get(seat, [])), known_tops[seat])

        for seat in seats:
            group = seat_to_group[seat]
            policy_name = seat_to_policy[seat]
            reward = float(rewards[seat])
            won = int(reward >= 1.0)
            tactical[group].update(trackers[seat].finish_game(env, bool(won)))
            policy_obj = policies[seat]
            if policy_name == "champion_cbp":
                prefix = "singleton" if group == "singleton" else "opponent"
                champion_head_stats[f"{prefix}_champion_chancellor"].update(policy_obj.chancellor_stats)
                champion_head_stats[f"{prefix}_champion_baron"].update(policy_obj.baron_stats)
                champion_head_stats[f"{prefix}_champion_prince"].update(policy_obj.prince_stats)
            records.append(
                {
                    "game": int(game),
                    "seed": int(seed),
                    "seat": seat,
                    "group": group,
                    "policy": policy_name,
                    "reward": reward,
                    "won": won,
                    "outcome": classify_outcome(seat, reward, elimination_order),
                    "singleton_seat": singleton_seat,
                    "starting_agent": starting_agent,
                }
            )

        if args.progress_every and (game + 1) % args.progress_every == 0:
            singleton_rows = [row for row in records if row["group"] == "singleton"]
            opponent_rows = [row for row in records if row["group"] == "opponents"]
            singleton_wr = sum(row["won"] for row in singleton_rows) / max(1, len(singleton_rows))
            opponent_wr = sum(row["won"] for row in opponent_rows) / max(1, len(opponent_rows))
            logger.write(
                f"Progression {name} {game + 1}/{args.games_per_matchup}",
                expected="Suivre le singleton et les trois adversaires.",
                actual={
                    "singleton_score_ge_1": round(float(singleton_wr), 4),
                    "opponent_copy_score_ge_1": round(float(opponent_wr), 4),
                },
            )

    singleton_rows = [row for row in records if row["group"] == "singleton"]
    opponent_rows = [row for row in records if row["group"] == "opponents"]
    return {
        "name": name,
        "singleton_policy": singleton_policy,
        "opponent_policy": opponent_policy,
        "singleton": summarize_group(singleton_rows, tactical["singleton"]),
        "opponents_per_copy": summarize_group(opponent_rows, tactical["opponents"]),
        "champion_head_stats": {
            key: {stat: int(value) for stat, value in sorted(counter.items())}
            for key, counter in champion_head_stats.items()
            if counter
        },
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Step6 - Matchups Asymetriques",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties par matchup: `{payload['args']['games_per_matchup']}`.",
        "",
        "Evaluation uniquement: un singleton joue contre trois copies d'un autre profil. "
        "Le siege du singleton tourne a chaque manche.",
        "",
        "## Synthese",
        "",
        "| Matchup | Singleton | Score >=1 | Victoire manche | Reward | Opposants/copie | Score >=1 | Victoire manche | Reward |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for result in payload["matchups"]:
        singleton = result["singleton"]
        opponents = result["opponents_per_copy"]
        lines.append(
            f"| {result['name']} | {POLICY_LABELS[result['singleton_policy']]} | "
            f"{pct(singleton['score_ge_1'])} | {pct(singleton['main_round_win_rate'])} | "
            f"{singleton['mean_reward']:.4f} | {POLICY_LABELS[result['opponent_policy']]} | "
            f"{pct(opponents['score_ge_1'])} | {pct(opponents['main_round_win_rate'])} | "
            f"{opponents['mean_reward']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Details Par Matchup",
            "",
        ]
    )
    for result in payload["matchups"]:
        singleton = result["singleton"]
        opponents = result["opponents_per_copy"]
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group_name, policy_name, row in [
            ("Singleton", result["singleton_policy"], singleton),
            ("Opposants/copie", result["opponent_policy"], opponents),
        ]:
            outcomes = row["outcomes"]
            lines.append(
                f"| {group_name} | {POLICY_LABELS[policy_name]} | {row['games']} | "
                f"{pct(row['score_ge_1'])} | +/- {pct(row['score_ge_1_ci95'])} | "
                f"{pct(row['main_round_win_rate'])} | {pct(row['spy_bonus_rate'])} | "
                f"{row['mean_reward']:.4f} | {pct(outcomes['first_out']['rate'])} | "
                f"{pct(outcomes['second_out']['rate'])} | {pct(outcomes['third_out']['rate'])} | "
                f"{pct(outcomes['final_loser']['rate'])} |"
            )
        lines.extend(
            [
                "",
                "| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group_name, row in [("Singleton", singleton), ("Opposants/copie", opponents)]:
            tact = row["tactical"]
            lines.append(
                f"| {group_name} | {pct(tact['guard_hit_rate'])} | {pct(tact['baron_win_rate'])} | "
                f"{pct(tact['baron_loss_rate'])} | {pct(tact['chancellor_keep_highest_rate'])} | "
                f"{pct(tact['priest_to_guard_hit_rate'])} |"
            )
        if result["champion_head_stats"]:
            lines.extend(
                [
                    "",
                    "Activite des tetes champion:",
                    "",
                    "```json",
                    json.dumps(result["champion_head_stats"], indent=2, ensure_ascii=False),
                    "```",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Lecture",
            "",
            "Ces matchups ne remplacent pas l'arena de lignage complete. Ils mesurent "
            "plutot si un profil singleton tient quand les trois autres joueurs ont "
            "tous le meme niveau/style. C'est un bon test d'exploitabilite avant "
            "d'entrainer en self-play.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


MATCHUP_INDEX = {name: idx for idx, (name, _singleton, _opponent) in enumerate(MATCHUPS)}


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate asymmetric lineage matchups.")
    parser.add_argument("--games-per-matchup", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=3500000)
    parser.add_argument("--matchup-seed-stride", type=int, default=50000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--chancellor-head", default="step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth")
    parser.add_argument("--chancellor-margin", type=float, default=0.10)
    parser.add_argument("--retarget-margin", type=float, default=0.10)
    parser.add_argument("--veto-score", type=float, default=0.05)
    parser.add_argument("--force-score", type=float, default=0.32)
    parser.add_argument("--self-force-score", type=float, default=0.55)
    parser.add_argument("--min-princess-prob", type=float, default=0.24)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--curriculum-checkpoint", default="curriculum_phase1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="asymmetric_matchups_5000_seed3500000.json")
    parser.add_argument("--markdown", default="asymmetric_matchups_5000_seed3500000.md")
    parser.add_argument("--run-log", default="step6_self_play/logs/2026-04-26_asymmetric_matchups_5000_seed3500000.md")
    parser.add_argument("--matchups", nargs="+", default=[name for name, _singleton, _opponent in MATCHUPS])
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step6 matchups asymetriques",
        expected="Tester champion/anciens modeles en singleton contre trois copies.",
        actual=f"games_per_matchup={args.games_per_matchup}, seed_start={args.seed_start}",
        details=vars(args),
    )

    contexts = {
        "step2_retarget": prepare_policy_context("step2_retarget", args),
        "step3_fast": prepare_policy_context("step3_fast_dagger", args),
        "curriculum_phase1": {"checkpoint": resolve_checkpoint(args.curriculum_checkpoint)},
    }
    chancellor_path = PROJECT_ROOT / args.chancellor_head
    contexts["chancellor_head"], _ckpt = load_chancellor_head(chancellor_path, args.device)

    results = []
    requested = set(args.matchups)
    unknown = requested - {name for name, _singleton, _opponent in MATCHUPS}
    if unknown:
        raise ValueError(f"Unknown matchup(s): {sorted(unknown)}")
    for name, singleton, opponent in MATCHUPS:
        if name not in requested:
            continue
        logger.write(
            f"Debut matchup {name}",
            expected=f"{POLICY_LABELS[singleton]} vs 3 {POLICY_LABELS[opponent]}",
        )
        result = evaluate_matchup(name, singleton, opponent, args, contexts, logger)
        results.append(result)
        logger.write(
            f"Fin matchup {name}",
            actual={
                "singleton_score_ge_1": result["singleton"]["score_ge_1"],
                "singleton_main_win": result["singleton"]["main_round_win_rate"],
                "opponent_copy_score_ge_1": result["opponents_per_copy"]["score_ge_1"],
                "opponent_copy_main_win": result["opponents_per_copy"]["main_round_win_rate"],
            },
        )

    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "matchups": results,
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step6 matchups asymetriques",
        expected="Produire rapport JSON + markdown.",
        actual=f"json={output}, markdown={markdown}",
        details={
            result["name"]: {
                "singleton_score_ge_1": result["singleton"]["score_ge_1"],
                "singleton_main_win": result["singleton"]["main_round_win_rate"],
                "opponent_copy_score_ge_1": result["opponents_per_copy"]["score_ge_1"],
                "opponent_copy_main_win": result["opponents_per_copy"]["main_round_win_rate"],
            }
            for result in results
        },
    )
    print(
        json.dumps(
            {
                result["name"]: {
                    "singleton_score_ge_1": result["singleton"]["score_ge_1"],
                    "singleton_main_win": result["singleton"]["main_round_win_rate"],
                    "opponent_copy_score_ge_1": result["opponents_per_copy"]["score_ge_1"],
                    "opponent_copy_main_win": result["opponents_per_copy"]["main_round_win_rate"],
                }
                for result in results
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
