"""Lineage arena before self-play training.

This evaluates four policies at the same table:

- champion_cbp: Step3 + Chancellor + Baron + Prince specialists
- step3_fast: Step3 fast DAgger policy
- step2_retarget: Step2 retarget distilled policy
- heuristic_fair: heuristic bot with randomized target tie-breaks

The result is evaluation-only. It is meant to decide whether this population is
healthy enough to become the first self-play league.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import permutations
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
    make_policy,
    prepare_policy_context,
    summarize_tactical,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp
from step5_execution_heads.chancellor_head import load_chancellor_head
from step5_execution_heads.evaluate_combined_three_heads import Step5ThreeSeat
from step5_execution_heads.cards.baron.evaluate_baron_specialist import direct_eliminations_from_event


STEP_DIR = PROJECT_ROOT / "step6_self_play"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"

POLICIES = ["champion_cbp", "step3_fast", "step2_retarget", "heuristic_fair"]

POLICY_LABELS = {
    "champion_cbp": "Champion CBP",
    "step3_fast": "Step3 seul",
    "step2_retarget": "Step2",
    "curriculum_phase1": "Curriculum phase1",
    "heuristic_fair": "Heuristique fair",
}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def ci95(winrate: float, n: int) -> float:
    return float(1.96 * np.sqrt(winrate * (1.0 - winrate) / max(1, n)))


def make_lineage_policy(policy_name: str, agent: str, args, contexts: dict):
    roles = {seat: "model" for seat in [f"player_{idx}" for idx in range(4)]}
    if policy_name == "heuristic_fair":
        return HeuristicBot(shuffle_targets=True)
    if policy_name == "step2_retarget":
        return make_policy("step2_retarget", args, roles, agent, contexts["step2_retarget"])
    if policy_name == "curriculum_phase1":
        return make_policy("step2_retarget", args, roles, agent, contexts["curriculum_phase1"])
    if policy_name == "step3_fast":
        return make_policy("step3_fast_dagger", args, roles, agent, contexts["step3_fast"])
    if policy_name == "champion_cbp":
        base = make_policy("step3_fast_dagger", args, roles, agent, contexts["step3_fast"])
        return Step5ThreeSeat(
            base,
            contexts["chancellor_head"],
            args,
            use_chancellor=True,
            use_baron=True,
            use_prince=True,
        )
    raise ValueError(policy_name)


def policy_action(policy_name: str, policy, env, obs_dict, agent: str) -> int:
    if policy_name == "heuristic_fair":
        return int(policy.choose_action(env, agent))
    return int(policy.act(env, obs_dict, agent))


def summarize_records(records: list[dict], tactical: dict[str, Counter]) -> dict:
    by_policy = {}
    for policy_name in POLICIES:
        rows = [record for record in records if record["policy"] == policy_name]
        wins = sum(row["won"] for row in rows)
        rewards = [row["reward"] for row in rows]
        outcomes = Counter(row["outcome"] for row in rows)
        seats = defaultdict(list)
        for row in rows:
            seats[row["seat"]].append(row)
        winrate = float(wins / max(1, len(rows)))
        by_policy[policy_name] = {
            "games": int(len(rows)),
            "wins": int(wins),
            "winrate": winrate,
            "winrate_ci95": ci95(winrate, len(rows)),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "outcomes": {
                key: {
                    "count": int(outcomes[key]),
                    "rate": float(outcomes[key] / max(1, len(rows))),
                }
                for key in ["winner", "first_out", "second_out", "third_out", "final_loser"]
            },
            "by_seat": {
                seat: {
                    "games": len(seat_rows),
                    "winrate": float(sum(row["won"] for row in seat_rows) / max(1, len(seat_rows))),
                }
                for seat, seat_rows in sorted(seats.items())
            },
            "tactical": summarize_tactical(tactical[policy_name]),
        }
    return by_policy


def evaluate(args, logger: ExperimentLogger) -> dict:
    contexts = {
        "step2_retarget": prepare_policy_context("step2_retarget", args),
        "step3_fast": prepare_policy_context("step3_fast_dagger", args),
    }
    chancellor_path = PROJECT_ROOT / args.chancellor_head
    contexts["chancellor_head"], _ckpt = load_chancellor_head(chancellor_path, args.device)

    env = LoveLetterRLEnv(num_players=4)
    seat_names = [f"player_{idx}" for idx in range(4)]
    assignments = list(permutations(POLICIES))
    records = []
    tactical = {policy_name: Counter() for policy_name in POLICIES}
    head_stats = {
        "champion_chancellor": Counter(),
        "champion_baron": Counter(),
        "champion_prince": Counter(),
    }

    for game in range(args.games):
        seed = args.seed_start + game
        np.random.seed(seed)
        assignment = assignments[game % len(assignments)]
        seat_to_policy = {seat: assignment[idx] for idx, seat in enumerate(seat_names)}
        starting_agent = seat_names[(game // len(assignments)) % len(seat_names)]
        env.reset(seed=seed, options={"starting_agent": starting_agent})
        policies = {
            agent: make_lineage_policy(policy_name, agent, args, contexts)
            for agent, policy_name in seat_to_policy.items()
        }
        trackers = {agent: TacticalTracker(agent) for agent in seat_names}
        rewards = {agent: 0.0 for agent in seat_names}
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
            pre_hands = {seat: list(env._hands.get(seat, [])) for seat in seat_names}
            known_tops = {seat: env._deck_knowledge.get(seat, {}).get(0) for seat in seat_names}
            trackers[agent].before_eval_action(env, event)
            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            for seat in seat_names:
                trackers[seat].observe_known_draw(pre_hands[seat], list(env._hands.get(seat, [])), known_tops[seat])

        for agent in seat_names:
            policy_name = seat_to_policy[agent]
            reward = float(rewards[agent])
            won = int(reward >= 1.0)
            tactical[policy_name].update(trackers[agent].finish_game(env, bool(won)))
            policy_obj = policies[agent]
            if policy_name == "champion_cbp":
                head_stats["champion_chancellor"].update(policy_obj.chancellor_stats)
                head_stats["champion_baron"].update(policy_obj.baron_stats)
                head_stats["champion_prince"].update(policy_obj.prince_stats)
            records.append(
                {
                    "game": int(game),
                    "seed": int(seed),
                    "seat": agent,
                    "policy": policy_name,
                    "reward": reward,
                    "won": won,
                    "outcome": classify_outcome(agent, reward, elimination_order),
                    "starting_agent": starting_agent,
                    "assignment": seat_to_policy,
                }
            )

        if args.progress_every and (game + 1) % args.progress_every == 0:
            partial = summarize_records(records, tactical)
            logger.write(
                f"Progression lineage arena {game + 1}/{args.games}",
                expected="Suivre les winrates pendant l'evaluation.",
                actual={name: round(partial[name]["winrate"], 4) for name in POLICIES},
            )

    return {
        "created_at": now_stamp(),
        "args": vars(args),
        "champion": {
            "base": args.step3_fast_checkpoint,
            "chancellor_head": args.chancellor_head,
            "baron": "step5_execution_heads/cards/baron/evaluate_baron_specialist.py",
            "prince": "step5_execution_heads/cards/prince/evaluate_prince_specialist.py",
        },
        "policies": summarize_records(records, tactical),
        "head_stats": {name: {key: int(value) for key, value in sorted(counter.items())} for name, counter in head_stats.items()},
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Step6 - Arena De Lignage",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}`.",
        "",
        "## Lecture",
        "",
        "Evaluation uniquement: ce benchmark ne met pas encore a jour les modeles. "
        "Il sert a verifier si le champion courant domine son lignage direct avant "
        "de construire une ligue de self-play.",
        "",
        "## Winrates",
        "",
        "| Politique | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward moyen | 1er sorti | 2e sorti | 3e sorti | Perd final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name in POLICIES:
        row = payload["policies"][policy_name]
        outcomes = row["outcomes"]
        tact_raw = row["tactical"]["raw_counts"]
        games = max(1, row["games"])
        lines.append(
            f"| {POLICY_LABELS[policy_name]} | {pct(row['winrate'])} | +/- {pct(row['winrate_ci95'])} | "
            f"{pct(tact_raw.get('main_round_wins', 0) / games)} | "
            f"{pct(tact_raw.get('spy_bonus_wins', 0) / games)} | "
            f"{row['mean_reward']:.4f} | {pct(outcomes['first_out']['rate'])} | "
            f"{pct(outcomes['second_out']['rate'])} | {pct(outcomes['third_out']['rate'])} | "
            f"{pct(outcomes['final_loser']['rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Winrate Par Siege",
            "",
            "| Politique | player_0 | player_1 | player_2 | player_3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy_name in POLICIES:
        row = payload["policies"][policy_name]["by_seat"]
        lines.append(
            f"| {POLICY_LABELS[policy_name]} | "
            f"{pct(row.get('player_0', {}).get('winrate', 0.0))} | "
            f"{pct(row.get('player_1', {}).get('winrate', 0.0))} | "
            f"{pct(row.get('player_2', {}).get('winrate', 0.0))} | "
            f"{pct(row.get('player_3', {}).get('winrate', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## Tactiques",
            "",
            "| Politique | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_name in POLICIES:
        tact = payload["policies"][policy_name]["tactical"]
        lines.append(
            f"| {POLICY_LABELS[policy_name]} | {pct(tact['guard_hit_rate'])} | "
            f"{pct(tact['baron_win_rate'])} | {pct(tact['baron_loss_rate'])} | "
            f"{pct(tact['chancellor_keep_highest_rate'])} | {pct(tact['priest_to_guard_hit_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Activite Des Tetes Champion",
            "",
            "```json",
            json.dumps(payload["head_stats"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Conclusion Provisoire",
            "",
            "Cette arena est le pont entre evaluation classique et self-play. Si le "
            "champion domine proprement cette table, la prochaine etape logique est "
            "de transformer ce lignage en population d'entrainement, avec conservation "
            "des anciens checkpoints pour eviter l'overfitting a un seul adversaire.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate champion against its lineage at the same table.")
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=3400000)
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
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="lineage_arena_5000_seed3400000.json")
    parser.add_argument("--markdown", default="lineage_arena_5000_seed3400000.md")
    parser.add_argument("--run-log", default="step6_self_play/logs/2026-04-26_lineage_arena_5000_seed3400000.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step6 lineage arena",
        expected="Comparer champion_cbp, Step3, Step2 et heuristique a la meme table.",
        actual=f"games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    payload = evaluate(args, logger)
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step6 lineage arena",
        expected="Produire un rapport de lignage avant self-play.",
        actual=f"json={output}, markdown={markdown}",
        details={name: payload["policies"][name]["winrate"] for name in POLICIES},
    )
    print(json.dumps({name: payload["policies"][name]["winrate"] for name in POLICIES}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
