"""Collect Step5 execution-teacher labels with paired CRN rollouts.

This script does not train a new model. It audits natural Step3 decisions:
when Step3 already chooses Priest/Baron/King or reaches a Chancellor choice,
we evaluate all legal executions of that same card/subdecision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import (
    CONFIG_HEURISTIC_COUNTS,
    CONFIG_LABELS,
    build_roles,
    decode_planned_event,
    evaluate_candidate_actions_paired_for_agent,
    make_policy,
    prepare_policy_context,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp
from step2_rl_finetune.evaluate_step2 import random_action
from step3_action_value.mini_rollout_probe import CARD_NAMES, decode_action
from step3_action_value.train_advantage_head_v2 import paired_delta_stats


STEP5_DIR = PROJECT_ROOT / "step5_execution_heads"
REPORT_DIR = STEP5_DIR / "reports"
LOG_DIR = STEP5_DIR / "logs"
DATASET_DIR = STEP5_DIR / "datasets"

TARGET_CARDS = {
    2: "priest_target",
    3: "baron_target",
    7: "king_target",
}

KIND_LABELS = {
    "priest_target": "Pretre - choix de cible",
    "baron_target": "Baron - choix de cible",
    "baron_low_target": "Baron avec carte faible - choix de cible",
    "king_target": "Roi - choix de cible",
    "chancellor_choice": "Chancelier - choix de carte/ordre",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR, DATASET_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def bounded_seed(value: int) -> int:
    return int(value % (2**32 - 1))


def finite_or_none(value):
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    return finite_or_none(value)


def phase_for_deck(deck_len: int) -> str:
    if deck_len >= 11:
        return "early"
    if deck_len >= 6:
        return "mid"
    return "late"


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def action_card(action: int) -> int:
    return int(action) // 100


def same_card_candidates(mask: np.ndarray, card: int) -> list[int]:
    return [action for action in valid_actions(mask) if action_card(action) == int(card)]


def chancellor_candidates(mask: np.ndarray) -> list[int]:
    return [action for action in valid_actions(mask) if int(action) >= 900]


def role_action_non_model(env, agent: str, obs_dict: dict, roles: dict[str, str], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


def determine_kind_and_candidates(env, agent: str, obs_dict: dict, action: int) -> tuple[str | None, list[int], dict]:
    mask = obs_dict["action_mask"]
    meta = {
        "hand": [int(card) for card in env._hands.get(agent, [])],
        "pool": [int(card) for card in getattr(env, "_chancellor_pool", [])],
        "deck_remaining": int(len(env._deck)),
        "phase": phase_for_deck(len(env._deck)),
    }
    if env._chancellor_pending and env.agent_selection == agent and int(action) >= 900:
        candidates = chancellor_candidates(mask)
        return "chancellor_choice", candidates, meta

    event = decode_planned_event(env, agent, int(action))
    if event["kind"] != "card":
        return None, [], meta
    card = int(event["card"])
    if card not in TARGET_CARDS:
        return None, [], meta
    candidates = same_card_candidates(mask, card)
    kind = TARGET_CARDS[card]
    if card == 3:
        remaining = event.get("remaining_hand", [])
        kept = int(remaining[0]) if remaining else None
        meta["baron_kept_card"] = kept
        if kept is not None and kept <= 4:
            kind = "baron_low_target"
    all_legal = valid_actions(mask)
    meta["legal_action_count"] = len(all_legal)
    meta["same_card_candidate_count"] = len(candidates)
    meta["other_card_legal"] = any(action_card(candidate) != card and candidate < 900 for candidate in all_legal)
    return kind, candidates, meta


def decision_seed_for(seed: int, turn: int, action: int, kind: str) -> int:
    kind_value = sum((idx + 1) * ord(ch) for idx, ch in enumerate(kind))
    return bounded_seed(seed * 1_000_003 + turn * 9176 + int(action) * 131 + kind_value)


def score_rows(
    env,
    eval_agent: str,
    roles: dict[str, str],
    candidates: list[int],
    original_action: int,
    base_checkpoint: Path,
    args,
    decision_seed: int,
) -> dict:
    rows, corr = evaluate_candidate_actions_paired_for_agent(
        env=env,
        eval_agent=eval_agent,
        actions=[int(action) for action in candidates],
        checkpoint=base_checkpoint,
        roles=roles,
        rollouts_per_action=args.rollouts_per_action,
        continuation=args.continuation,
        reward_score_weight=args.reward_score_weight,
        decision_seed=decision_seed,
    )
    by_action = {int(row["action"]): row for row in rows}
    model_row = by_action.get(int(original_action))
    best_row = rows[0] if rows else None
    delta = None
    clear_regret = False
    if model_row is not None and best_row is not None:
        delta = paired_delta_stats(best_row, model_row)
        clear_regret = (
            int(best_row["action"]) != int(original_action)
            and delta["mean_win_delta"] >= args.min_win_delta
            and delta["mean_score_delta"] >= args.min_score_delta
            and delta["t_stat"] >= args.min_t_stat
        )
    return {
        "rows": rows,
        "crn_correlation": corr,
        "best_row": best_row,
        "model_row": model_row,
        "delta_best_vs_model": delta,
        "clear_regret": clear_regret,
    }


def should_collect(kind: str, config_name: str, counts: Counter, pair_counts: Counter, args) -> bool:
    if args.states_per_config_kind > 0:
        return pair_counts[(kind, config_name)] < args.states_per_config_kind
    return counts[kind] < args.max_states_per_kind


def all_targets_met(counts: Counter, pair_counts: Counter, target_kinds: set[str], args) -> bool:
    if args.states_per_config_kind > 0:
        return all(
            pair_counts[(kind, config_name)] >= args.states_per_config_kind
            for kind in target_kinds
            for config_name in CONFIG_HEURISTIC_COUNTS
        )
    return all(counts[kind] >= args.max_states_per_kind for kind in target_kinds)


def summarize_records(records: list[dict], args) -> dict:
    by_kind = defaultdict(list)
    for record in records:
        by_kind[record["kind"]].append(record)

    summary = {}
    for kind, rows in sorted(by_kind.items()):
        clear = [row for row in rows if row["clear_regret"]]
        forced_execution = [row for row in rows if row["candidate_count"] <= 1]
        forced_card = [row for row in rows if not row.get("other_card_legal", True)]
        deltas = [
            row["delta_best_vs_model"]["mean_score_delta"]
            for row in rows
            if row.get("delta_best_vs_model") and row["delta_best_vs_model"]["mean_score_delta"] is not None
        ]
        win_deltas = [
            row["delta_best_vs_model"]["mean_win_delta"]
            for row in rows
            if row.get("delta_best_vs_model") and row["delta_best_vs_model"]["mean_win_delta"] is not None
        ]
        best_diff = [
            row
            for row in rows
            if row.get("best_action") is not None and int(row["best_action"]) != int(row["model_action"])
        ]
        summary[kind] = {
            "label": KIND_LABELS.get(kind, kind),
            "states": int(len(rows)),
            "clear_regret_states": int(len(clear)),
            "clear_regret_rate": float(len(clear) / max(1, len(rows))),
            "best_differs_from_model": int(len(best_diff)),
            "best_differs_rate": float(len(best_diff) / max(1, len(rows))),
            "forced_execution_states": int(len(forced_execution)),
            "forced_execution_rate": float(len(forced_execution) / max(1, len(rows))),
            "no_other_card_legal_states": int(len(forced_card)),
            "no_other_card_legal_rate": float(len(forced_card) / max(1, len(rows))),
            "mean_score_regret": float(np.mean(deltas)) if deltas else 0.0,
            "mean_win_regret": float(np.mean(win_deltas)) if win_deltas else 0.0,
            "examples_clear_regret": clean_json(clear[:8]),
        }
    return {
        "thresholds": {
            "rollouts_per_action": args.rollouts_per_action,
            "min_win_delta": args.min_win_delta,
            "min_score_delta": args.min_score_delta,
            "min_t_stat": args.min_t_stat,
        },
        "total_states": int(len(records)),
        "by_kind": summary,
    }


def write_markdown(payload: dict, path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Step5 - Execution Teacher Initial",
        "",
        f"Date: {payload['created_at']}.",
        "",
        "Objectif: mesurer le regret oracle des executions Roi/Baron/Pretre/Chancelier dans les etats naturels du Step3 rapide.",
        "",
        "## Parametres",
        "",
        f"- Games par composition: `{payload['args']['games']}`",
        f"- Max states par type: `{payload['args']['max_states_per_kind']}`",
        f"- Rollouts CRN par action: `{payload['args']['rollouts_per_action']}`",
        f"- Continuation rollout: `{payload['args']['continuation']}`",
        f"- Seuil regret clair: win_delta >= `{payload['args']['min_win_delta']}`, score_delta >= `{payload['args']['min_score_delta']}`, t >= `{payload['args']['min_t_stat']}`",
        "",
        "## Synthese",
        "",
        "| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, row in summary["by_kind"].items():
        lines.append(
            "| {label} | {states} | {best_diff} ({best_rate}) | {clear} ({clear_rate}) | {forced} ({forced_rate}) | {card_forced} ({card_forced_rate}) | {score:.4f} | {win:.4f} |".format(
                label=row["label"],
                states=row["states"],
                best_diff=row["best_differs_from_model"],
                best_rate=pct(row["best_differs_rate"]),
                clear=row["clear_regret_states"],
                clear_rate=pct(row["clear_regret_rate"]),
                forced=row["forced_execution_states"],
                forced_rate=pct(row["forced_execution_rate"]),
                card_forced=row["no_other_card_legal_states"],
                card_forced_rate=pct(row["no_other_card_legal_rate"]),
                score=row["mean_score_regret"],
                win=row["mean_win_regret"],
            )
        )

    lines.extend(
        [
            "",
            "## Lecture",
            "",
            "- `Best != modele` signifie que l'oracle rollout prefere une autre execution en moyenne.",
            "- `Regret clair` applique les seuils statistiques stricts ci-dessus.",
            "- `Execution forcee` signifie qu'il n'y avait qu'une seule execution legale de cette carte.",
            "- `Carte forcee` signifie qu'au moment ou la carte a ete jouee, le modele n'avait pas d'autre carte/action principale legale.",
            "",
            "## Prochaine Etape",
            "",
            "Les types avec assez de `regret clair` deviennent candidats pour un dataset d'entrainement. Les autres doivent rester en audit ou etre collectes avec plus de rollouts/etats.",
            "",
            "## Fichiers",
            "",
            f"- Dataset JSON: `{payload['dataset_path']}`",
            f"- Rapport JSON: `{payload['json_path']}`",
            f"- Log: `{payload['run_log']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(args, logger: ExperimentLogger) -> list[dict]:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    context_args = SimpleNamespace(**vars(args))
    context = prepare_policy_context("step3_fast_dagger", context_args)
    target_kinds = set(args.kinds)
    records = []
    counts = Counter()
    pair_counts = Counter()

    for config_idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
        seed_base = args.seed_start + config_idx * args.seed_stride
        logger.write(
            f"Collecte config {config_name}",
            expected="Collecter des etats naturels Step3 puis scorer les executions.",
            actual=f"{CONFIG_LABELS[config_name]}, seed_base={seed_base}",
        )
        for game in range(args.games):
            if all_targets_met(counts, pair_counts, target_kinds, args):
                break
            seed = seed_base + game
            np.random.seed(seed)
            env.reset(seed=seed)
            eval_agent = f"player_{game % 4}"
            roles = build_roles(eval_agent, heuristic_count, game)
            policies = {eval_agent: make_policy("step3_fast_dagger", context_args, roles, eval_agent, context)}

            for turn, agent in enumerate(env.agent_iter()):
                obs_dict, reward, terminated, truncated, _info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue

                if agent == eval_agent:
                    action = int(policies[agent].act(env, obs_dict, agent))
                    kind, candidates, meta = determine_kind_and_candidates(env, agent, obs_dict, action)
                    if kind in target_kinds and should_collect(kind, config_name, counts, pair_counts, args):
                        decision_seed = decision_seed_for(seed, turn, action, kind)
                        if len(candidates) > 0:
                            scored = score_rows(
                                env,
                                eval_agent,
                                roles,
                                candidates,
                                action,
                                context["base_checkpoint"],
                                args,
                                decision_seed,
                            )
                            best = scored["best_row"]
                            delta = clean_json(scored["delta_best_vs_model"])
                            record = {
                                "kind": kind,
                                "label": KIND_LABELS.get(kind, kind),
                                "config": config_name,
                                "seed": int(seed),
                                "turn": int(turn),
                                "seat": eval_agent,
                                "phase": meta["phase"],
                                "deck_remaining": meta["deck_remaining"],
                                "observation": obs_dict["observation"].astype(np.float32),
                                "action_mask": obs_dict["action_mask"].astype(np.int8),
                                "hand": meta["hand"],
                                "pool": meta["pool"],
                                "baron_kept_card": meta.get("baron_kept_card"),
                                "model_action": int(action),
                                "model_decoded": decode_action(int(action)),
                                "best_action": int(best["action"]) if best else None,
                                "best_decoded": decode_action(int(best["action"])) if best else None,
                                "candidate_count": int(len(candidates)),
                                "other_card_legal": bool(meta.get("other_card_legal", True)),
                                "candidate_rows": clean_json(scored["rows"]),
                                "crn_correlation": clean_json(scored["crn_correlation"]),
                                "delta_best_vs_model": delta,
                                "clear_regret": bool(scored["clear_regret"]),
                                "decision_seed": int(decision_seed),
                            }
                            records.append(record)
                            counts[kind] += 1
                            pair_counts[(kind, config_name)] += 1
                            if counts[kind] % max(1, args.progress_every) == 0:
                                logger.write(
                                    f"Collecte {kind}: {counts[kind]}/{args.max_states_per_kind}",
                                    expected="Progression par type.",
                                    actual=(
                                        f"dernier seed={seed}, phase={meta['phase']}, "
                                        f"config_count={pair_counts[(kind, config_name)]}"
                                    ),
                                )
                    env.step(action)
                else:
                    action = role_action_non_model(env, agent, obs_dict, roles, bot)
                    env.step(action)
        logger.write(
            f"Fin config {config_name}",
            expected="Passer a la composition suivante ou terminer si quotas atteints.",
            actual={"counts": dict(counts), "pair_counts": {f"{k}|{c}": v for (k, c), v in pair_counts.items()}},
        )
        if all_targets_met(counts, pair_counts, target_kinds, args):
            break
    return records


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Collect Step5 execution-teacher labels.")
    parser.add_argument("--games", type=int, default=300, help="Games per composition scan.")
    parser.add_argument("--seed-start", type=int, default=510000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--max-states-per-kind", type=int, default=40)
    parser.add_argument(
        "--states-per-config-kind",
        type=int,
        default=0,
        help="If >0, collect this many states for each kind and arena composition.",
    )
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--reward-score-weight", type=float, default=0.05)
    parser.add_argument("--min-win-delta", type=float, default=0.10)
    parser.add_argument("--min-score-delta", type=float, default=0.05)
    parser.add_argument("--min-t-stat", type=float, default=0.75)
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["priest_target", "baron_target", "baron_low_target", "king_target", "chancellor_choice"],
        choices=list(KIND_LABELS),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default="execution_teacher_initial.json")
    parser.add_argument("--output", default="execution_teacher_initial_report.json")
    parser.add_argument("--markdown", default="execution_teacher_initial_report.md")
    parser.add_argument("--run-log", default="step5_execution_heads/logs/2026-04-25_execution_teacher_initial.md")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step5 execution teacher",
        expected="Mesurer le regret oracle sur Roi/Baron/Pretre/Chancelier.",
        actual=f"kinds={args.kinds}, max={args.max_states_per_kind}, rollouts={args.rollouts_per_action}",
        details=vars(args),
    )

    records = collect(args, logger)
    summary = summarize_records(records, args)

    dataset_path = DATASET_DIR / args.dataset
    json_path = REPORT_DIR / args.output
    markdown_path = REPORT_DIR / args.markdown
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "records": clean_json(records),
        "summary": clean_json(summary),
        "dataset_path": str(dataset_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "run_log": args.run_log,
    }
    dataset_path.write_text(json.dumps(payload["records"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_markdown(payload, markdown_path)

    logger.write(
        "Fin Step5 execution teacher",
        expected="Obtenir un audit initial et un dataset brut.",
        actual=f"states={len(records)}, json={json_path}, markdown={markdown_path}",
        details=summary,
    )
    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
