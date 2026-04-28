"""Promotion gate for Step7 self-play candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_rl_finetune.common import ExperimentLogger, now_stamp
from step7_self_play_league.league_policy import (
    DEFAULT_PROMOTION_PATH,
    DEFAULT_ROSTER_PATH,
    append_jsonl,
    apply_promotion,
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


def metric(payload: dict, policy_id: str, key: str, default: float = 0.0) -> float:
    return float(payload["policies"].get(policy_id, {}).get(key, default))


def tactical_metric(payload: dict, policy_id: str, key: str, default: float = 0.0) -> float:
    return float(payload["policies"].get(policy_id, {}).get("tactical", {}).get(key, default))


def promotion_decision(args, roster: dict, payload: dict) -> dict:
    candidate_id = args.candidate_id
    policies = policy_by_id(roster)
    if candidate_id not in policies:
        raise ValueError(f"Candidate missing from roster: {candidate_id}")
    if candidate_id not in payload["policies"]:
        raise ValueError(f"Candidate missing from evaluation report: {candidate_id}")

    active_before = [policy for policy in roster["policies"] if policy.get("active", False)]
    best_active = max(active_before, key=lambda item: float(item.get("elo", 1500.0)))
    champion = policies.get(args.champion_id, best_active)
    candidate_elo = metric(payload, candidate_id, "elo")
    best_elo = metric(payload, best_active["policy_id"], "elo", float(best_active.get("elo", 1500.0)))
    candidate_main = metric(payload, candidate_id, "main_round_win_rate")
    best_main = metric(payload, best_active["policy_id"], "main_round_win_rate")
    candidate_guard = tactical_metric(payload, candidate_id, "guard_hit_rate")
    champion_guard = tactical_metric(payload, champion["policy_id"], "guard_hit_rate")
    candidate_baron_loss = tactical_metric(payload, candidate_id, "baron_loss_rate")
    champion_baron_loss = tactical_metric(payload, champion["policy_id"], "baron_loss_rate")
    candidate_chancellor = tactical_metric(payload, candidate_id, "chancellor_keep_highest_rate")
    champion_chancellor = tactical_metric(payload, champion["policy_id"], "chancellor_keep_highest_rate")

    checks = {
        "elo": candidate_elo >= best_elo + args.min_elo_gain,
        "main_win_guardrail": candidate_main >= best_main - args.main_win_tolerance,
        "baron_loss_guardrail": candidate_baron_loss <= champion_baron_loss + args.max_baron_loss_increase,
        "chancellor_guardrail": candidate_chancellor >= champion_chancellor - args.max_chancellor_drop,
        "guard_hit_guardrail": candidate_guard >= champion_guard - args.max_guard_drop,
    }
    accepted = bool(args.force or all(checks.values()))
    return {
        "candidate_id": candidate_id,
        "best_active_id": best_active["policy_id"],
        "champion_id": champion["policy_id"],
        "accepted": accepted,
        "forced": bool(args.force),
        "checks": checks,
        "metrics": {
            "candidate_elo": candidate_elo,
            "best_elo": best_elo,
            "candidate_main_round_win_rate": candidate_main,
            "best_main_round_win_rate": best_main,
            "candidate_guard_hit_rate": candidate_guard,
            "champion_guard_hit_rate": champion_guard,
            "candidate_baron_loss_rate": candidate_baron_loss,
            "champion_baron_loss_rate": champion_baron_loss,
            "candidate_chancellor_keep_highest_rate": candidate_chancellor,
            "champion_chancellor_keep_highest_rate": champion_chancellor,
        },
    }


def write_report(decision: dict, path: Path) -> None:
    lines = [
        "# Step7 - Promotion Decision",
        "",
        f"Date: {decision['created_at']}.",
        "",
        f"Candidate: `{decision['candidate_id']}`.",
        f"Decision: `{'PROMOTED' if decision['accepted'] else 'REJECTED'}`.",
        f"Dropped: `{decision.get('dropped_id')}`.",
        "",
        "## Checks",
        "",
    ]
    for key, value in decision["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Metrics", ""])
    for key, value in decision["metrics"].items():
        lines.append(f"- `{key}`: {value:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Promote or reject a Step7 candidate from an Elo report.")
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER_PATH))
    parser.add_argument("--promotion-history", default=str(DEFAULT_PROMOTION_PATH))
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--champion-id", default="champion_cbp")
    parser.add_argument("--min-elo-gain", type=float, default=20.0)
    parser.add_argument("--main-win-tolerance", type=float, default=0.01)
    parser.add_argument("--max-baron-loss-increase", type=float, default=0.05)
    parser.add_argument("--max-chancellor-drop", type=float, default=0.08)
    parser.add_argument("--max-guard-drop", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-log", default="step7_self_play_league/logs/league_promote.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    roster = load_roster(args.roster)
    payload = json.loads(Path(args.evaluation_report).read_text(encoding="utf-8"))
    decision = promotion_decision(args, roster, payload)
    decision["created_at"] = now_stamp()
    dropped_id = None
    if decision["accepted"] and not args.dry_run:
        policies = policy_by_id(roster)
        for policy_id, row in payload["policies"].items():
            if policy_id in policies:
                policies[policy_id]["elo"] = float(row.get("elo", policies[policy_id].get("elo", 1500.0)))
                policies[policy_id]["games"] = int(row.get("games", policies[policy_id].get("games", 0)))
        roster, dropped_id = apply_promotion(roster, args.candidate_id)
        save_roster(roster, args.roster)
    decision["dropped_id"] = dropped_id
    decision["dry_run"] = bool(args.dry_run)
    append_jsonl(args.promotion_history, {"event": "promotion_decision", **decision})
    output = REPORT_DIR / (args.output or f"{args.candidate_id}_promotion_decision.md")
    write_report(decision, output)
    logger.write(
        "Decision promotion Step7",
        actual=f"accepted={decision['accepted']}, dropped={dropped_id}, dry_run={args.dry_run}",
        details=decision,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
