"""Conditional card-execution ablations for the current Step3 fast policy.

We keep the model's high-level decision to play a card, then randomize only the
fine execution of that card: target, Guard guess, or Chancellor choice. This
separates "the model knows when to play the card" from "the model knows how to
execute the card once chosen".
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
    TacticalTracker,
    build_roles,
    classify_outcome,
    decode_planned_event,
    json_safe,
    make_policy,
    prepare_policy_context,
    summarize_tactical,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step2_rl_finetune.evaluate_step2 import random_action, summarize_rewards
from step3_action_value.mini_rollout_probe import CARD_NAMES, decode_action


STEP4_DIR = PROJECT_ROOT / "step4_weakness_analysis"
REPORT_DIR = STEP4_DIR / "reports"
LOG_DIR = STEP4_DIR / "logs"

ABLATION_LABELS = {
    "baseline": "Step3 rapide normal",
    "guard_target_random": "Garde cible random, guess conserve",
    "guard_guess_random": "Garde guess random, cible conservee",
    "guard_target_guess_random": "Garde cible+guess random",
    "priest_target_random": "Pretre cible random",
    "baron_target_random": "Baron cible random",
    "prince_target_random": "Prince cible random",
    "king_target_random": "Roi cible random",
    "chancellor_choice_random": "Chancelier choix random",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def bounded_seed(value: int) -> int:
    return int(value % (2**32 - 1))


def stable_name_value(name: str) -> int:
    total = 0
    for idx, char in enumerate(name):
        total += (idx + 1) * ord(char)
    return total


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def action_card(action: int) -> int:
    return int(action) // 100


def action_target(action: int) -> int:
    return (int(action) % 100) // 10


def action_guess(action: int) -> int:
    return int(action) % 10


def choose_random(candidates: list[int], seed: int) -> int:
    if not candidates:
        raise ValueError("No candidates to sample.")
    rng = np.random.default_rng(bounded_seed(seed))
    return int(rng.choice(np.asarray(candidates, dtype=np.int64)))


def same_card_candidates(mask: np.ndarray, card: int) -> list[int]:
    return [action for action in valid_actions(mask) if action_card(action) == card]


def ablate_action(
    ablation: str,
    env: LoveLetterRLEnv,
    agent: str,
    obs_dict: dict,
    original_action: int,
    seed: int,
    turn: int,
) -> tuple[int, dict]:
    """Return possibly-randomized action and intervention metadata."""

    meta = {
        "eligible": False,
        "randomizable": False,
        "changed": False,
        "ablation": ablation,
        "original_action": int(original_action),
        "new_action": int(original_action),
        "original_decoded": decode_action(int(original_action)),
        "new_decoded": decode_action(int(original_action)),
        "candidate_count": 0,
    }
    if ablation == "baseline":
        return int(original_action), meta

    mask = obs_dict["action_mask"]
    sample_seed = seed * 1_000_003 + turn * 9176 + stable_name_value(ablation)

    if ablation == "chancellor_choice_random":
        if env._chancellor_pending and env.agent_selection == agent and int(original_action) >= 900:
            candidates = [action for action in valid_actions(mask) if action >= 900]
            meta["eligible"] = True
            meta["candidate_count"] = len(candidates)
            meta["randomizable"] = len(candidates) > 1
            if candidates:
                new_action = choose_random(candidates, sample_seed)
                meta["changed"] = int(new_action) != int(original_action)
                meta["new_action"] = int(new_action)
                meta["new_decoded"] = decode_action(int(new_action))
                return int(new_action), meta
        return int(original_action), meta

    event = decode_planned_event(env, agent, int(original_action))
    if event["kind"] != "card":
        return int(original_action), meta
    card = int(event["card"])

    candidates: list[int] = []
    if ablation == "guard_target_random" and card == 1:
        meta["eligible"] = True
        guess = action_guess(original_action)
        candidates = [
            action
            for action in same_card_candidates(mask, 1)
            if action_guess(action) == guess
        ]
    elif ablation == "guard_guess_random" and card == 1:
        meta["eligible"] = True
        target = action_target(original_action)
        candidates = [
            action
            for action in same_card_candidates(mask, 1)
            if action_target(action) == target
        ]
    elif ablation == "guard_target_guess_random" and card == 1:
        meta["eligible"] = True
        candidates = same_card_candidates(mask, 1)
    elif ablation == "priest_target_random" and card == 2:
        meta["eligible"] = True
        candidates = same_card_candidates(mask, 2)
    elif ablation == "baron_target_random" and card == 3:
        meta["eligible"] = True
        candidates = same_card_candidates(mask, 3)
    elif ablation == "prince_target_random" and card == 5:
        meta["eligible"] = True
        candidates = same_card_candidates(mask, 5)
    elif ablation == "king_target_random" and card == 7:
        meta["eligible"] = True
        candidates = same_card_candidates(mask, 7)

    if not meta["eligible"]:
        return int(original_action), meta

    meta["candidate_count"] = len(candidates)
    meta["randomizable"] = len(candidates) > 1
    if not candidates:
        return int(original_action), meta
    new_action = choose_random(candidates, sample_seed)
    meta["changed"] = int(new_action) != int(original_action)
    meta["new_action"] = int(new_action)
    meta["new_decoded"] = decode_action(int(new_action))
    return int(new_action), meta


def direct_eliminations_from_event(event: dict, agent: str) -> list[str]:
    if event["kind"] != "card":
        return []
    card = event["card"]
    target = event.get("target")
    target_card = event.get("target_card")
    eliminated = []
    if card == 1 and target and target_card == event.get("guess"):
        eliminated.append(target)
    elif card == 3 and target:
        my_after = event.get("remaining_hand", [])
        my_val = my_after[0] if my_after else None
        if my_val is not None and target_card is not None:
            if my_val > target_card:
                eliminated.append(target)
            elif target_card > my_val:
                eliminated.append(agent)
    elif card == 5 and target and target_card == 9:
        eliminated.append(target)
    elif card == 9:
        eliminated.append(agent)
    return eliminated


def summarize_outcomes(records: list[dict]) -> dict:
    rewards = [record["reward"] for record in records]
    wins = [record["won"] for record in records]
    summary = summarize_rewards(rewards, wins)
    outcome_counts = Counter(record["outcome"] for record in records)
    summary["outcomes"] = {
        key: {
            "count": int(outcome_counts[key]),
            "rate": float(outcome_counts[key] / max(1, len(records))),
        }
        for key in ["winner", "first_out", "second_out", "third_out", "final_loser", "unknown_loss"]
    }
    losses = [record for record in records if not record["won"]]
    loss_counts = Counter(record["outcome"] for record in losses)
    summary["loss_positions"] = {
        key: {
            "count": int(loss_counts[key]),
            "rate_among_losses": float(loss_counts[key] / max(1, len(losses))),
        }
        for key in ["first_out", "second_out", "third_out", "final_loser", "unknown_loss"]
    }
    by_seat = {}
    for seat in [f"player_{idx}" for idx in range(4)]:
        seat_records = [record for record in records if record["seat"] == seat]
        if seat_records:
            by_seat[seat] = summarize_rewards(
                [record["reward"] for record in seat_records],
                [record["won"] for record in seat_records],
            )
    summary["by_seat"] = by_seat
    summary["games"] = int(len(records))
    return summary


def summarize_interventions(records: list[dict]) -> dict:
    eligible_records = [record for record in records if record["eligible_events"] > 0]
    changed_records = [record for record in records if record["changed_events"] > 0]
    counts = Counter()
    examples = []
    for record in records:
        counts["eligible_events"] += record["eligible_events"]
        counts["randomizable_events"] += record["randomizable_events"]
        counts["changed_events"] += record["changed_events"]
        counts["decisions"] += record["decisions"]
        for key, value in record["candidate_counts"].items():
            counts[f"candidate_count_{key}"] += value
        if len(examples) < 12:
            examples.extend(record["examples"][: 12 - len(examples)])
    return {
        "eligible_games": int(len(eligible_records)),
        "eligible_game_rate": float(len(eligible_records) / max(1, len(records))),
        "changed_games": int(len(changed_records)),
        "changed_game_rate": float(len(changed_records) / max(1, len(records))),
        "eligible_events": int(counts["eligible_events"]),
        "randomizable_events": int(counts["randomizable_events"]),
        "changed_events": int(counts["changed_events"]),
        "decision_count": int(counts["decisions"]),
        "eligible_events_per_game": float(counts["eligible_events"] / max(1, len(records))),
        "changed_events_per_game": float(counts["changed_events"] / max(1, len(records))),
        "changed_per_eligible_event": float(counts["changed_events"] / max(1, counts["eligible_events"])),
        "eligible_summary": summarize_outcomes(eligible_records) if eligible_records else None,
        "changed_summary": summarize_outcomes(changed_records) if changed_records else None,
        "examples": examples,
    }


def role_action_non_model(env, agent: str, obs_dict: dict, roles: dict[str, str], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


def evaluate_ablation_config(
    ablation: str,
    config_name: str,
    games: int,
    seed_start: int,
    args,
    context: dict,
    logger: ExperimentLogger,
) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_tactical = Counter()

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policies = {eval_agent: make_policy("step3_fast_dagger", args, roles, eval_agent, context)}
        tracker = TacticalTracker(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        interventions = Counter()
        candidate_counts = Counter()
        examples = []

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            if agent == eval_agent:
                original_action = int(policies[agent].act(env, obs_dict, agent))
                action, meta = ablate_action(ablation, env, agent, obs_dict, original_action, seed, turn)
                interventions["decisions"] += 1
                if meta["eligible"]:
                    interventions["eligible_events"] += 1
                if meta["randomizable"]:
                    interventions["randomizable_events"] += 1
                if meta["changed"]:
                    interventions["changed_events"] += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "turn": int(turn),
                                "seed": int(seed),
                                "seat": eval_agent,
                                "original": meta["original_decoded"],
                                "new": meta["new_decoded"],
                                "candidate_count": int(meta["candidate_count"]),
                            }
                        )
                if meta["eligible"]:
                    candidate_counts[str(meta["candidate_count"])] += 1
            else:
                action = role_action_non_model(env, agent, obs_dict, roles, bot)

            event = decode_planned_event(env, agent, action)
            pre_eval_hand = list(env._hands.get(eval_agent, []))
            known_top = env._deck_knowledge.get(eval_agent, {}).get(0)

            if agent == eval_agent:
                tracker.before_eval_action(env, event)

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        records.append(
            {
                "seed": int(seed),
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "elimination_order": elimination_order,
                "eligible_events": int(interventions["eligible_events"]),
                "randomizable_events": int(interventions["randomizable_events"]),
                "changed_events": int(interventions["changed_events"]),
                "decisions": int(interventions["decisions"]),
                "candidate_counts": dict(candidate_counts),
                "examples": examples,
            }
        )

    summary = summarize_outcomes(records)
    summary["interventions"] = summarize_interventions(records)
    summary["tactical"] = summarize_tactical(aggregate_tactical)
    logger.write(
        f"{ablation} termine {config_name}",
        expected="Ablation conditionnelle sur le Step3 rapide.",
        actual=(
            f"winrate={summary['winrate']:.4f}, "
            f"changed_events={summary['interventions']['changed_events']}, "
            f"changed_games={summary['interventions']['changed_games']}"
        ),
    )
    return summary


def evaluate_ablation(ablation: str, args, context: dict, logger: ExperimentLogger) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{ablation} - {config_name}",
            expected="Meme arena fair seat-rotated que l'interlude post-fix.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        configs[config_name] = evaluate_ablation_config(
            ablation,
            config_name,
            args.games,
            seed_start,
            args,
            context,
            logger,
        )
    return {
        "label": ABLATION_LABELS[ablation],
        "configs": configs,
        "composite": composite_score(configs),
    }


def aggregate_interventions(policy: dict) -> dict:
    total_records = sum(config["games"] for config in policy["configs"].values())
    counts = Counter()
    eligible_summaries = []
    changed_summaries = []
    for config in policy["configs"].values():
        inter = config["interventions"]
        for key in [
            "eligible_games",
            "changed_games",
            "eligible_events",
            "randomizable_events",
            "changed_events",
            "decision_count",
        ]:
            counts[key] += inter[key]
        if inter["eligible_summary"]:
            eligible_summaries.append(inter["eligible_summary"])
        if inter["changed_summary"]:
            changed_summaries.append(inter["changed_summary"])
    return {
        "eligible_games": int(counts["eligible_games"]),
        "eligible_game_rate": float(counts["eligible_games"] / max(1, total_records)),
        "changed_games": int(counts["changed_games"]),
        "changed_game_rate": float(counts["changed_games"] / max(1, total_records)),
        "eligible_events": int(counts["eligible_events"]),
        "randomizable_events": int(counts["randomizable_events"]),
        "changed_events": int(counts["changed_events"]),
        "changed_per_eligible_event": float(counts["changed_events"] / max(1, counts["eligible_events"])),
    }


def tactical_totals(policy: dict) -> dict:
    total = Counter()
    for config in policy["configs"].values():
        total.update(config["tactical"]["raw_counts"])
    return summarize_tactical(total)


def heuristic_read(delta: float, interventions: dict) -> str:
    changed_games = interventions["changed_games"]
    if changed_games < 50:
        return "Signal rare: ne pas conclure sans plus de parties."
    if delta <= -0.015:
        return "Competence d'execution probable: le hasard degrade nettement."
    if delta >= 0.010:
        return "Alerte: le hasard ameliore, execution modele suspecte."
    if abs(delta) <= 0.005:
        return "Effet global faible: verifier le regret oracle avant d'entrainer."
    if delta < 0:
        return "Competence moderee ou effet rare."
    return "Effet legerement positif: investiguer avant correction."


def write_markdown(payload: dict, path: Path) -> None:
    baseline = payload["ablations"]["baseline"]
    baseline_comp = baseline["composite"]
    lines = [
        "# Step4 - Ablations Conditionnelles Par Carte",
        "",
        f"Date: {payload['created_at']}.",
        "",
        "Objectif: comprendre ce qui vient du style du modele et ce qui vient de la dynamique du jeu.",
        "",
        (
            "On laisse Step3 rapide choisir naturellement quelle carte jouer. "
            "Si la carte jouee correspond a l'ablation, on randomise seulement "
            "son execution: cible, guess, ou choix Chancelier."
        ),
        "",
        "## Synthese Globale",
        "",
        "| Ablation | Composite | Delta vs normal | Changed games | Changed events | Lecture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, policy in payload["ablations"].items():
        inter = aggregate_interventions(policy)
        delta = policy["composite"] - baseline_comp
        lines.append(
            "| {label} | {comp:.5f} | {delta:+.5f} | {changed_games} | {changed_events} | {read} |".format(
                label=policy["label"],
                comp=policy["composite"],
                delta=delta,
                changed_games=inter["changed_games"],
                changed_events=inter["changed_events"],
                read=heuristic_read(delta, inter) if name != "baseline" else "Reference normale.",
            )
        )

    lines.extend(
        [
            "",
            "## Winrates Par Composition",
            "",
            "| Ablation | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["ablations"].items():
        configs = policy["configs"]
        lines.append(
            "| {label} | {a} | {b} | {c} | {d} |".format(
                label=policy["label"],
                a=pct(configs["vs_0H_3R"]["winrate"]),
                b=pct(configs["vs_1H_2R"]["winrate"]),
                c=pct(configs["vs_2H_1R"]["winrate"]),
                d=pct(configs["vs_3H"]["winrate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interventions",
            "",
            "`Eligible` signifie que le modele a choisi la carte concernee. `Changed` signifie que le tirage random a effectivement remplace son action.",
            "",
            "| Ablation | Eligible games | Changed games | Eligible events | Changed events | Changed / eligible |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["ablations"].items():
        inter = aggregate_interventions(policy)
        lines.append(
            "| {label} | {eg} | {cg} | {ee} | {ce} | {ratio} |".format(
                label=policy["label"],
                eg=inter["eligible_games"],
                cg=inter["changed_games"],
                ee=inter["eligible_events"],
                ce=inter["changed_events"],
                ratio=pct(inter["changed_per_eligible_event"]),
            )
        )

    lines.extend(
        [
            "",
            "## Metriques Tactiques Globales",
            "",
            "| Ablation | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier connu gagne |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["ablations"].items():
        tact = tactical_totals(policy)
        lines.append(
            "| {label} | {guard} | {known_guard} | {priest_guard} | {baron_win} | {baron_loss} | {chancellor} |".format(
                label=policy["label"],
                guard=pct(tact["guard_hit_rate"]),
                known_guard=pct(tact["known_guard_hit_rate"]),
                priest_guard=pct(tact["priest_to_guard_hit_rate"]),
                baron_win=pct(tact["baron_win_rate"]),
                baron_loss=pct(tact["baron_loss_rate"]),
                chancellor=pct(tact["chancellor_known_draw_win_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Comment Lire Ces Resultats",
            "",
            "- Si une ablation degrade fort, le modele a une competence d'execution a proteger.",
            "- Si une ablation ne change presque rien, on doit verifier le regret oracle: soit la carte est peu sensible, soit le modele ne sait pas l'exploiter.",
            "- Si une ablation ameliore, c'est une alerte: l'execution du modele est probablement toxique sur cette carte.",
            "- Le delta global doit toujours etre lu avec le nombre de `changed events`; une competence rare peut etre importante sans bouger beaucoup le composite.",
            "",
            "## Fichiers",
            "",
            f"- JSON: `{payload['json_path']}`",
            f"- Log: `{payload['run_log']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Conditional card-execution ablation arena for Step3 fast.")
    parser.add_argument("--games", type=int, default=1000, help="Games per composition and ablation.")
    parser.add_argument("--seed-start", type=int, default=260000)
    parser.add_argument("--seed-stride", type=int, default=10000)
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
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_LABELS))
    parser.add_argument("--output", default="step3_fast_card_ablation_1000.json")
    parser.add_argument("--markdown", default="step3_fast_card_ablation_1000.md")
    parser.add_argument("--run-log", default="step4_weakness_analysis/logs/2026-04-25_step3_fast_card_ablation_1000.md")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.ablations if name not in ABLATION_LABELS]
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")
    if "baseline" not in args.ablations:
        args.ablations = ["baseline"] + list(args.ablations)

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step4 ablations conditionnelles",
        expected="Quantifier la maitrise fine des cartes par rapport au hasard.",
        actual=f"games={args.games}, ablations={args.ablations}",
        details=vars(args),
    )

    context_args = SimpleNamespace(**vars(args))
    context = prepare_policy_context("step3_fast_dagger", context_args)
    ablations = {}
    for ablation in args.ablations:
        logger.write(
            f"Debut ablation {ablation}",
            expected=ABLATION_LABELS[ablation],
            actual="Evaluation sur les quatre compositions.",
        )
        ablations[ablation] = evaluate_ablation(ablation, context_args, context, logger)
        logger.write(
            f"Fin ablation {ablation}",
            expected="Comparer au baseline ensuite.",
            actual=f"composite={ablations[ablation]['composite']:.5f}",
        )

    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    payload = {
        "created_at": now_stamp(),
        "checkpoint": str(context["checkpoint"]),
        "base_checkpoint": str(context["base_checkpoint"]),
        "games_per_config": args.games,
        "seed_start": args.seed_start,
        "seat_rotation": "eval_agent = player_{game % 4}",
        "heuristic_mode": "shuffle_targets=True",
        "ablations": ablations,
        "json_path": str(output),
        "markdown_path": str(markdown),
        "run_log": args.run_log,
    }
    payload = json_safe(payload)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step4 ablations conditionnelles",
        expected="Produire un rapport d'ablation exploitable.",
        actual=f"json={output}, markdown={markdown}",
    )

    summary = {
        name: {
            "composite": policy["composite"],
            "delta_vs_baseline": policy["composite"] - ablations["baseline"]["composite"],
            "interventions": aggregate_interventions(policy),
        }
        for name, policy in ablations.items()
    }
    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
