"""Compare HeuristicBot, Step2, and the two active Step3 branches.

The goal of this interlude is intentionally narrow: answer whether the current
neural policies are actually weaker than the heuristic teacher when evaluated
on the same arena compositions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import (
    OPPONENT_CONFIGS,
    evaluate_player0_heuristic,
    evaluate_player0_model,
)
from step3_action_value.evaluate_advantage_head_v2 import (
    evaluate_player0_advantage,
    load_advantage_bundle,
)


INTERLUDE_DIR = PROJECT_ROOT / "interlude_heuristic_comparison"
REPORT_DIR = INTERLUDE_DIR / "reports"
LOG_DIR = INTERLUDE_DIR / "logs"
STEP3_REPORT_DIR = PROJECT_ROOT / "step3_action_value" / "reports"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def summarize_policy_config(seed_rows: list[dict]) -> dict:
    games = sum(int(row["games"]) for row in seed_rows)
    wins = sum(int(row["wins"]) for row in seed_rows)
    winrate = wins / games if games else 0.0
    mean_reward = (
        sum(float(row["mean_reward"]) * int(row["games"]) for row in seed_rows) / games
        if games
        else 0.0
    )
    return {
        "games": games,
        "wins": wins,
        "winrate": winrate,
        "winrate_ci95": 1.96 * (winrate * (1.0 - winrate) / max(1, games)) ** 0.5,
        "mean_reward": mean_reward,
        "seeds": [row.get("seed_start") for row in seed_rows],
    }


def aggregate_policy(seed_results: dict[int, dict]) -> dict:
    configs = {}
    for config in OPPONENT_CONFIGS:
        rows = []
        for seed, result in seed_results.items():
            row = dict(result[config])
            row["seed_start"] = seed
            rows.append(row)
        configs[config] = summarize_policy_config(rows)
    return {"configs": configs, "composite": composite_score(configs)}


def diff_ci95(row_a: dict, row_b: dict) -> float:
    pa = float(row_a["winrate"])
    pb = float(row_b["winrate"])
    na = max(1, int(row_a["games"]))
    nb = max(1, int(row_b["games"]))
    return 1.96 * ((pa * (1.0 - pa) / na) + (pb * (1.0 - pb) / nb)) ** 0.5


def composite_diff_ci95(configs_a: dict, configs_b: dict) -> float:
    weights = {"vs_0H_3R": 0.10, "vs_1H_2R": 0.20, "vs_2H_1R": 0.30, "vs_3H": 0.40}
    variance = 0.0
    for name, weight in weights.items():
        pa = float(configs_a[name]["winrate"])
        pb = float(configs_b[name]["winrate"])
        na = max(1, int(configs_a[name]["games"]))
        nb = max(1, int(configs_b[name]["games"]))
        variance += (weight**2) * ((pa * (1.0 - pa) / na) + (pb * (1.0 - pb) / nb))
    return 1.96 * variance**0.5


def load_hybrid_report(seed: int) -> dict:
    path = STEP3_REPORT_DIR / f"step3_advantage_v2_attempt2_strict_eval_1000_m010_verify16_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))["model_configs"]


def build_advantage_args(args, verify: bool) -> SimpleNamespace:
    return SimpleNamespace(
        max_actions=args.max_actions,
        override_margin=args.override_margin,
        entropy_margin_scale=0.0,
        min_deck_progress=0.0,
        min_played_cards=0,
        max_active_players=4,
        verify_rollouts=args.verify_rollouts if verify else 0,
        verify_min_win_delta=args.verify_min_win_delta,
        verify_min_score_delta=args.verify_min_score_delta,
        verify_t_threshold=args.verify_t_threshold,
        verify_player0_continuation="heuristic",
        verify_reward_score_weight=0.05,
        device=args.device,
        example_limit=0,
    )


def evaluate_step3_fast(args, seed: int, logger: ExperimentLogger) -> dict:
    checkpoint, base_checkpoint, head, ckpt = load_advantage_bundle(args.step3_fast_checkpoint, None)
    categories = ckpt.get("categories", [])
    eval_args = build_advantage_args(args, verify=False)
    if eval_args.max_actions is None:
        eval_args.max_actions = int(ckpt.get("max_actions", 14))

    output = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        logger.write(
            f"Step3 rapide {name} seed={seed}",
            expected="Mesurer la tete autonome DAgger sans verify.",
            actual=f"checkpoint={checkpoint}",
        )
        output[name] = evaluate_player0_advantage(
            base_checkpoint,
            head,
            categories,
            opponents,
            args.games,
            seed,
            eval_args,
        )
    return output


def evaluate_step2(args, seed: int, logger: ExperimentLogger) -> dict:
    checkpoint = resolve_checkpoint(args.step2_checkpoint)
    output = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        logger.write(
            f"Step2 {name} seed={seed}",
            expected="Mesurer le socle neural actuel contre la meme composition.",
            actual=f"checkpoint={checkpoint}",
        )
        output[name] = evaluate_player0_model(checkpoint, opponents, args.games, seed)
    return output


def evaluate_heuristic(args, seed: int, logger: ExperimentLogger) -> dict:
    output = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        logger.write(
            f"HeuristicBot {name} seed={seed}",
            expected="Mesurer le professeur heuristique dans la meme arena.",
            actual=f"games={args.games}",
        )
        output[name] = evaluate_player0_heuristic(opponents, args.games, seed)
    return output


def comparison_table(policies: dict) -> list[dict]:
    heuristic = policies["heuristic"]["configs"]
    rows = []
    for policy_name, payload in policies.items():
        row = {
            "policy": policy_name,
            "composite": payload["composite"],
            "delta_vs_heuristic": payload["composite"] - policies["heuristic"]["composite"],
            "delta_vs_heuristic_ci95_approx": composite_diff_ci95(payload["configs"], heuristic),
        }
        for config in OPPONENT_CONFIGS:
            row[config] = payload["configs"][config]["winrate"]
            row[f"{config}_delta_vs_heuristic"] = (
                payload["configs"][config]["winrate"] - heuristic[config]["winrate"]
            )
            row[f"{config}_delta_ci95_approx"] = diff_ci95(payload["configs"][config], heuristic[config])
        rows.append(row)
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    rows = payload["comparison_rows"]
    names = {
        "heuristic": "HeuristicBot",
        "step2_retarget": "Step2 retarget",
        "step3_fast_dagger": "Step3 rapide DAgger",
        "step3_hybrid_verify16": "Step3 hybride verify16",
    }
    lines = [
        "# Interlude - Heuristique vs Step2 vs Step3",
        "",
        f"Date: {payload['created_at']}.",
        "",
        "Objectif: verifier proprement si les modeles actuels sont vraiment moins bons que `HeuristicBot`.",
        "",
        f"Protocole: {payload['games_per_seed_config']} parties par composition et par seed, seeds {payload['seeds']}.",
        "La Step3 hybride `verify16` reutilise les rapports officiels deja calcules sur ces memes seeds.",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Heuristic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {r0:.2%} | {r1:.2%} | {r2:.2%} | {r3:.2%} | {comp:.5f} | {delta:+.5f} |".format(
                name=names.get(row["policy"], row["policy"]),
                r0=row["vs_0H_3R"],
                r1=row["vs_1H_2R"],
                r2=row["vs_2H_1R"],
                r3=row["vs_3H"],
                comp=row["composite"],
                delta=row["delta_vs_heuristic"],
            )
        )
    lines.extend(
        [
            "",
            "## Lecture",
            "",
            "- Un delta est considere clair seulement s'il depasse grossierement son intervalle d'incertitude approximatif.",
            "- La question centrale est donc: les modeles Step2/Step3 sont-ils sous `HeuristicBot`, au meme niveau, ou au-dessus ?",
            "",
        ]
    )
    for row in rows:
        if row["policy"] == "heuristic":
            continue
        lines.append(
            "- `{}`: delta composite vs HeuristicBot = `{:+.5f}` ; IC95 approx du delta = `+/- {:.5f}`.".format(
                names.get(row["policy"], row["policy"]),
                row["delta_vs_heuristic"],
                row["delta_vs_heuristic_ci95_approx"],
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            payload["conclusion"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_conclusion(rows: list[dict]) -> str:
    by_name = {row["policy"]: row for row in rows}
    lines = []
    for name in ["step2_retarget", "step3_fast_dagger", "step3_hybrid_verify16"]:
        row = by_name[name]
        ci = row["delta_vs_heuristic_ci95_approx"]
        delta = row["delta_vs_heuristic"]
        if delta > ci:
            verdict = "au-dessus de l'heuristique de facon nette"
        elif delta < -ci:
            verdict = "en-dessous de l'heuristique de facon nette"
        else:
            verdict = "statistiquement proche de l'heuristique sur ce test"
        lines.append(f"`{name}` est {verdict}: delta composite `{delta:+.5f}`, IC95 approx `+/- {ci:.5f}`.")
    return " ".join(lines)


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Interlude arena: HeuristicBot vs Step2 vs Step3 branches.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[134000, 135000, 136000])
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=16)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="interlude_arena_heuristic_step2_step3_3x1000.json")
    parser.add_argument("--markdown", default="interlude_arena_heuristic_step2_step3_3x1000.md")
    parser.add_argument("--run-log", default="interlude_heuristic_comparison/logs/2026-04-25_interlude_arena.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut interlude arena",
        expected="Comparer HeuristicBot, Step2, Step3 rapide et Step3 hybride sur les memes seeds.",
        actual=f"games={args.games}, seeds={args.seeds}",
        details=vars(args),
    )

    raw = {
        "heuristic": {},
        "step2_retarget": {},
        "step3_fast_dagger": {},
        "step3_hybrid_verify16": {},
    }
    for seed in args.seeds:
        raw["heuristic"][seed] = evaluate_heuristic(args, seed, logger)
        raw["step2_retarget"][seed] = evaluate_step2(args, seed, logger)
        raw["step3_fast_dagger"][seed] = evaluate_step3_fast(args, seed, logger)
        raw["step3_hybrid_verify16"][seed] = load_hybrid_report(seed)
        logger.write(
            f"Seed {seed} termine",
            expected="Avoir les quatre politiques pour ce seed.",
            actual="hybride verify16 reutilise depuis les rapports Step3 officiels.",
        )

    policies = {name: aggregate_policy(seed_results) for name, seed_results in raw.items()}
    rows = comparison_table(policies)
    conclusion = make_conclusion(rows)
    payload = {
        "created_at": now_stamp(),
        "games_per_seed_config": args.games,
        "seeds": args.seeds,
        "total_games_per_config_policy": args.games * len(args.seeds),
        "notes": [
            "Step3 hybrid verify16 is loaded from official reports on the same seed starts.",
            "Composite weights: 0H=0.10, 1H=0.20, 2H=0.30, 3H=0.40.",
        ],
        "raw_by_seed": raw,
        "policies": policies,
        "comparison_rows": rows,
        "conclusion": conclusion,
    }

    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin interlude arena",
        expected="Conclusion explicite sur modeles vs HeuristicBot.",
        actual=conclusion,
        details={"json": str(output), "markdown": str(markdown), "comparison_rows": rows},
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
