"""Elo evaluator for the Step7 self-play league."""

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
    summarize_tactical,
)
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp
from step5_execution_heads.cards.baron.evaluate_baron_specialist import direct_eliminations_from_event
from step7_self_play_league.league_policy import (
    DEFAULT_RESULTS_PATH,
    DEFAULT_ROSTER_PATH,
    LeaguePolicyFactory,
    LeagueRuntimeArgs,
    active_policies,
    append_jsonl,
    load_roster,
    policy_by_id,
    save_roster,
)


STEP_DIR = PROJECT_ROOT / "step7_self_play_league"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def elo_expected(r_a: float, r_b: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0)))


def pair_score(reward_a: float, reward_b: float) -> float:
    if reward_a > reward_b:
        return 1.0
    if reward_a < reward_b:
        return 0.0
    return 0.5


def update_elo_pairwise(elos: dict[str, float], policy_ids: list[str], rewards: dict[str, float], k_factor: float) -> dict[str, float]:
    deltas = {policy_id: 0.0 for policy_id in policy_ids}
    for i, left in enumerate(policy_ids):
        for right in policy_ids[i + 1 :]:
            score_left = pair_score(rewards[left], rewards[right])
            expected_left = elo_expected(elos[left], elos[right])
            delta = k_factor * (score_left - expected_left)
            deltas[left] += delta
            deltas[right] -= delta
    for policy_id, delta in deltas.items():
        elos[policy_id] = float(elos[policy_id] + delta)
    return elos


def choose_table(active: list[dict], game_idx: int, rng: np.random.Generator) -> list[dict]:
    if len(active) < 4:
        raise ValueError("Need at least 4 active policies for league evaluation.")
    if len(active) == 4:
        chosen = list(active)
    else:
        indices = rng.choice(len(active), size=4, replace=False)
        chosen = [active[int(idx)] for idx in indices]
    # Rotate the chosen order deterministically so no policy owns a seat shape.
    offset = game_idx % 4
    return chosen[offset:] + chosen[:offset]


def summarize_policy(records: list[dict], tactical: Counter) -> dict:
    wins = sum(record["score_ge_1"] for record in records)
    rewards = [record["reward"] for record in records]
    outcomes = Counter(record["outcome"] for record in records)
    by_seat = defaultdict(list)
    for record in records:
        by_seat[record["seat"]].append(record)
    tact = summarize_tactical(tactical)
    raw = tact["raw_counts"]
    games = max(1, len(records))
    return {
        "games": int(len(records)),
        "score_ge_1": float(wins / games),
        "main_round_win_rate": float(raw.get("main_round_wins", 0) / games),
        "spy_bonus_rate": float(raw.get("spy_bonus_wins", 0) / games),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "outcomes": {
            key: {"count": int(outcomes[key]), "rate": float(outcomes[key] / games)}
            for key in ["winner", "first_out", "second_out", "third_out", "final_loser"]
        },
        "by_seat": {
            seat: {
                "games": len(rows),
                "score_ge_1": float(sum(row["score_ge_1"] for row in rows) / max(1, len(rows))),
            }
            for seat, rows in sorted(by_seat.items())
        },
        "tactical": tact,
    }


def evaluate(args, logger: ExperimentLogger) -> dict:
    roster = load_roster(args.roster)
    include_policies = list(getattr(args, "include_policy", []) or [])
    if include_policies:
        policies_by_id = policy_by_id(roster)
        for policy_id in include_policies:
            if policy_id not in policies_by_id:
                raise ValueError(f"Policy not found for temporary evaluation: {policy_id}")
            policies_by_id[policy_id]["active"] = True
    if args.bootstrap:
        for policy in roster["policies"]:
            policy["elo"] = 1500.0
            policy["games"] = 0

    policies = policy_by_id(roster)
    elos = {policy_id: float(policy.get("elo", 1500.0)) for policy_id, policy in policies.items()}
    rng = np.random.default_rng(args.seed)
    factory = LeaguePolicyFactory(
        LeagueRuntimeArgs(
            device=args.device,
            override_margin=args.override_margin,
            max_actions=args.max_actions,
            verify_rollouts=args.verify_rollouts,
            chancellor_margin=args.chancellor_margin,
            retarget_margin=args.retarget_margin,
            veto_score=args.veto_score,
            force_score=args.force_score,
            self_force_score=args.self_force_score,
            min_princess_prob=args.min_princess_prob,
            example_limit=args.example_limit,
        )
    )

    env = LoveLetterRLEnv(num_players=4)
    seats = [f"player_{idx}" for idx in range(4)]
    records = []
    tactical = {policy["policy_id"]: Counter() for policy in roster["policies"]}

    for game_idx in range(args.games):
        seed = args.seed_start + game_idx
        np.random.seed(seed)
        active = active_policies(roster)
        table = choose_table(active, game_idx, rng)
        seat_specs = {seat: table[idx] for idx, seat in enumerate(seats)}
        seat_policy_ids = {seat: spec["policy_id"] for seat, spec in seat_specs.items()}
        roles = {seat: "model" for seat in seats}
        starting_agent = seats[(game_idx // 4) % 4]
        env.reset(seed=seed, options={"starting_agent": starting_agent})
        seat_policies = {
            seat: factory.make(spec, seat, roles)
            for seat, spec in seat_specs.items()
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
            action = int(seat_policies[agent].act(env, obs_dict, agent))
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

        rewards_by_policy = {}
        for seat in seats:
            policy_id = seat_policy_ids[seat]
            reward = float(rewards[seat])
            rewards_by_policy[policy_id] = reward
            won = int(reward >= 1.0)
            tactical[policy_id].update(trackers[seat].finish_game(env, bool(won)))
            records.append(
                {
                    "game": int(game_idx),
                    "seed": int(seed),
                    "seat": seat,
                    "policy_id": policy_id,
                    "reward": reward,
                    "score_ge_1": won,
                    "outcome": classify_outcome(seat, reward, elimination_order),
                    "starting_agent": starting_agent,
                    "table": seat_policy_ids,
                }
            )
        update_elo_pairwise(elos, list(rewards_by_policy), rewards_by_policy, args.k_factor)
        for policy_id in rewards_by_policy:
            policies[policy_id]["games"] = int(policies[policy_id].get("games", 0)) + 1
            policies[policy_id]["elo"] = float(elos[policy_id])

        if not args.no_append_results:
            append_jsonl(
                args.results_jsonl,
                {
                    "event": "game",
                    "created_at": now_stamp(),
                    "game": int(game_idx),
                    "seed": int(seed),
                    "table": seat_policy_ids,
                    "rewards": {seat: float(value) for seat, value in rewards.items()},
                    "round_winners": list(getattr(env, "_round_winners", [])),
                    "spy_winner": getattr(env, "_round_spy_winner", None),
                    "elimination_order": elimination_order,
                },
            )

        if args.progress_every and (game_idx + 1) % args.progress_every == 0:
            logger.write(
                f"Progression Elo {game_idx + 1}/{args.games}",
                actual={policy_id: round(elos[policy_id], 1) for policy_id in sorted(elos)},
            )

    by_policy = {}
    for policy_id in sorted(policies):
        rows = [record for record in records if record["policy_id"] == policy_id]
        by_policy[policy_id] = summarize_policy(rows, tactical.get(policy_id, Counter()))
        by_policy[policy_id]["elo"] = float(elos[policy_id])
        by_policy[policy_id]["active"] = bool(policies[policy_id].get("active", False))

    if args.update_roster:
        save_roster(roster, args.roster)

    return {
        "created_at": now_stamp(),
        "args": vars(args),
        "policies": by_policy,
        "roster": roster,
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Step7 - League Elo Evaluation",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Games: `{payload['args']['games']}`.",
        "",
        "| Policy | Active | Elo | Games | Score >=1 | Main win | Spy bonus | Reward | Baron win | Baron loss | Chancellor max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def sort_key(item):
        _policy_id, row = item
        return (int(row["games"] > 0), float(row["elo"]))

    for policy_id, row in sorted(payload["policies"].items(), key=sort_key, reverse=True):
        tact = row["tactical"]
        lines.append(
            f"| {policy_id} | {row['active']} | {row['elo']:.1f} | {row['games']} | "
            f"{pct(row['score_ge_1'])} | {pct(row['main_round_win_rate'])} | "
            f"{pct(row['spy_bonus_rate'])} | {row['mean_reward']:.4f} | "
            f"{pct(tact['baron_win_rate'])} | {pct(tact['baron_loss_rate'])} | "
            f"{pct(tact['chancellor_keep_highest_rate'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate the Step7 active league with pairwise Elo.")
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER_PATH))
    parser.add_argument("--results-jsonl", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--seed-start", type=int, default=7100000)
    parser.add_argument("--k-factor", type=float, default=8.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--include-policy", action="append", default=[])
    parser.add_argument("--update-roster", action="store_true")
    parser.add_argument("--no-append-results", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--chancellor-margin", type=float, default=0.10)
    parser.add_argument("--retarget-margin", type=float, default=0.10)
    parser.add_argument("--veto-score", type=float, default=0.05)
    parser.add_argument("--force-score", type=float, default=0.32)
    parser.add_argument("--self-force-score", type=float, default=0.55)
    parser.add_argument("--min-princess-prob", type=float, default=0.24)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--output", default="league_eval.json")
    parser.add_argument("--markdown", default="league_eval.md")
    parser.add_argument("--run-log", default="step7_self_play_league/logs/league_eval.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut evaluation Elo Step7",
        expected="Evaluer la population active et mettre a jour Elo si demande.",
        actual=f"games={args.games}, k={args.k_factor}, roster={args.roster}",
        details=vars(args),
    )
    payload = evaluate(args, logger)
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin evaluation Elo Step7",
        actual=f"json={output}, markdown={markdown}",
        details={policy_id: row["elo"] for policy_id, row in payload["policies"].items()},
    )
    print(json.dumps({policy_id: row["elo"] for policy_id, row in payload["policies"].items()}, indent=2))


if __name__ == "__main__":
    main()
