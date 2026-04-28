"""Cluster the latest Step3 fast policy by card/phase archetypes.

The goal is deliberately diagnostic, not training. We rotate the evaluated
Step3 policy across seats, use fair heuristic opponents, and record which card
families appear in the model's own decisions during early/mid/late phases.
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
    GenericAdvantageSeat,
    TacticalTracker,
    build_roles,
    classify_outcome,
    decode_planned_event,
    json_safe,
    make_policy,
    prepare_policy_context,
    role_action,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step2_rl_finetune.evaluate_step2 import summarize_rewards
from step3_action_value.mini_rollout_probe import CARD_NAMES


STEP4_DIR = PROJECT_ROOT / "step4_weakness_analysis"
REPORT_DIR = STEP4_DIR / "reports"
LOG_DIR = STEP4_DIR / "logs"

PHASES = ["early", "mid", "late"]

CARD_GROUPS = {
    "information_active": {1, 2, 3, 6, 7},
    "hypothesis_pressure": {1, 3, 5, 7},
    "elimination_pressure": {1, 3, 5, 9},
    "hand_deck_control": {5, 6, 7, 8},
    "safe_tempo": {0, 4, 8},
    "passive_value_or_constraint": {0, 8, 9},
    "high_risk_trap": {3, 5, 7, 8, 9},
    "public_reveal": {1, 3, 5, 9},
}

CARD_GROUP_LABELS = {
    "information_active": "Information active",
    "hypothesis_pressure": "Hypothese / ciblage",
    "elimination_pressure": "Pression elimination",
    "hand_deck_control": "Controle main/pioche",
    "safe_tempo": "Tempo sur",
    "passive_value_or_constraint": "Valeur passive / contrainte",
    "high_risk_trap": "Risque fort",
    "public_reveal": "Revelation publique",
}

CLUSTER_LABELS = {
    "early_high_card": "Grosse carte tot",
    "early_princess": "Princesse tot",
    "early_countess": "Comtesse tot",
    "guard_heavy": "Beaucoup de Gardes",
    "baron_pressure": "Pression Baron",
    "low_baron_risk": "Baron avec petite carte",
    "priest_to_guard_line": "Pretre puis Garde",
    "chancellor_game": "Partie avec Chancelier",
    "spy_game": "Partie avec Espionne",
    "king_game": "Partie avec Roi",
    "prince_game": "Partie avec Prince",
    "handmaid_game": "Partie avec Servante",
    "info_dense": "Main riche en information",
    "control_dense": "Main riche en controle",
    "late_control": "Controle en fin de manche",
    "voluntary_countess": "Comtesse volontaire",
    "trap_dense": "Beaucoup de cartes pieges",
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def phase_for_deck(deck_len: int) -> str:
    if deck_len >= 11:
        return "early"
    if deck_len >= 6:
        return "mid"
    return "late"


def sorted_card_names(cards: set[int]) -> str:
    return ", ".join(CARD_NAMES[c] for c in sorted(cards))


def loss_position_summary(records: list[dict]) -> dict:
    losses = [record for record in records if not record["won"]]
    counts = Counter(record["outcome"] for record in losses)
    total = max(1, len(losses))
    return {
        key: {
            "count": int(counts[key]),
            "rate_among_losses": float(counts[key] / total),
        }
        for key in ["first_out", "second_out", "third_out", "final_loser", "unknown_loss"]
    }


def summarize_records(records: list[dict]) -> dict:
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
    summary["loss_positions"] = loss_position_summary(records)
    summary["games"] = int(len(records))
    return summary


def merge_counter_dict(counter: Counter) -> dict:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


class CardArchetypeTracker:
    """Tracks only the evaluated Step3 player's own hand/actions."""

    def __init__(self, eval_agent: str):
        self.eval_agent = eval_agent
        self.decisions = 0
        self.phase_decisions = Counter()
        self.card_phase_decisions = Counter()
        self.card_phase_games = set()
        self.group_phase_decisions = Counter()
        self.group_phase_games = set()
        self.played_card_phase = Counter()
        self.played_card_total = Counter()
        self.group_decisions_total = Counter()
        self.card_decisions_total = Counter()
        self.flags = Counter()
        self.current_phase = None
        self.played_priest_targets = set()
        self.priest_targets_seen = set()

    def _cards_visible_to_self(self, env) -> list[int]:
        if env._chancellor_pending and env.agent_selection == self.eval_agent:
            return list(env._chancellor_pool)
        return list(env._hands.get(self.eval_agent, []))

    def before_decision(self, env) -> None:
        phase = phase_for_deck(len(env._deck))
        self.current_phase = phase
        self.decisions += 1
        self.phase_decisions[phase] += 1

        hand = self._cards_visible_to_self(env)
        unique_cards = set(int(card) for card in hand)
        for card in unique_cards:
            self.card_phase_decisions[(card, phase)] += 1
            self.card_phase_games.add((card, phase))
            self.card_decisions_total[card] += 1
        for group, cards in CARD_GROUPS.items():
            if unique_cards & cards:
                self.group_phase_decisions[(group, phase)] += 1
                self.group_phase_games.add((group, phase))
                self.group_decisions_total[group] += 1

        if phase == "early" and any(card >= 7 for card in unique_cards):
            self.flags["early_high_card"] = 1
        if phase == "early" and 9 in unique_cards:
            self.flags["early_princess"] = 1
        if phase == "early" and 8 in unique_cards:
            self.flags["early_countess"] = 1
        if 3 in unique_cards:
            self.flags["baron_seen"] += 1
            other_cards = [card for card in hand if card != 3]
            if other_cards and max(other_cards) >= 5:
                self.flags["baron_pressure"] = 1
            if other_cards and max(other_cards) <= 4:
                self.flags["low_baron_risk"] = 1
        if phase == "late" and unique_cards & CARD_GROUPS["hand_deck_control"]:
            self.flags["late_control"] = 1
        if unique_cards & CARD_GROUPS["high_risk_trap"]:
            self.flags["trap_decisions"] += 1

    def after_action(self, event: dict) -> None:
        phase = self.current_phase or "unknown"
        if event["kind"] == "chancellor_choice":
            self.flags["chancellor_game"] = 1
            return

        card = int(event["card"])
        self.played_card_phase[(card, phase)] += 1
        self.played_card_total[card] += 1

        if card == 0:
            self.flags["spy_game"] = 1
        elif card == 1:
            self.flags["guard_plays"] += 1
            target = event.get("target")
            if target in self.priest_targets_seen:
                self.flags["priest_to_guard_line"] = 1
        elif card == 2:
            self.flags["priest_plays"] += 1
            target = event.get("target")
            if target:
                self.priest_targets_seen.add(target)
        elif card == 3:
            self.flags["baron_plays"] += 1
            my_after = event.get("remaining_hand", [])
            if my_after and my_after[0] <= 4:
                self.flags["low_baron_risk"] = 1
            if my_after and my_after[0] >= 5:
                self.flags["baron_pressure"] = 1
        elif card == 4:
            self.flags["handmaid_game"] = 1
        elif card == 5:
            self.flags["prince_game"] = 1
        elif card == 6:
            self.flags["chancellor_game"] = 1
        elif card == 7:
            self.flags["king_game"] = 1
        elif card == 8 and not event.get("forced_countess"):
            self.flags["voluntary_countess"] = 1

    def clusters(self) -> set[str]:
        clusters = set()
        for key in [
            "early_high_card",
            "early_princess",
            "early_countess",
            "baron_pressure",
            "low_baron_risk",
            "priest_to_guard_line",
            "chancellor_game",
            "spy_game",
            "king_game",
            "prince_game",
            "handmaid_game",
            "voluntary_countess",
            "late_control",
        ]:
            if self.flags.get(key):
                clusters.add(key)
        if self.flags["guard_plays"] >= 2 or self.card_decisions_total[1] >= 3:
            clusters.add("guard_heavy")
        if self.group_decisions_total["information_active"] >= 3:
            clusters.add("info_dense")
        if self.group_decisions_total["hand_deck_control"] >= 3:
            clusters.add("control_dense")
        if self.flags["trap_decisions"] >= 3:
            clusters.add("trap_dense")
        return clusters

    def finish(self) -> dict:
        return {
            "decisions": int(self.decisions),
            "phase_decisions": merge_counter_dict(self.phase_decisions),
            "card_phase_decisions": {
                f"{CARD_NAMES[card]}|{phase}": int(value)
                for (card, phase), value in sorted(self.card_phase_decisions.items())
            },
            "group_phase_decisions": {
                f"{group}|{phase}": int(value)
                for (group, phase), value in sorted(self.group_phase_decisions.items())
            },
            "played_card_phase": {
                f"{CARD_NAMES[card]}|{phase}": int(value)
                for (card, phase), value in sorted(self.played_card_phase.items())
            },
            "played_card_total": {
                CARD_NAMES[card]: int(value)
                for card, value in sorted(self.played_card_total.items())
            },
            "flags": merge_counter_dict(self.flags),
            "clusters": sorted(self.clusters()),
            "_card_phase_game_keys": sorted(self.card_phase_games),
            "_group_phase_game_keys": sorted(self.group_phase_games),
        }


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


def evaluate_config(config_name: str, games: int, seed_start: int, args, context: dict, logger: ExperimentLogger) -> dict:
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
        tactical = TacticalTracker(eval_agent)
        tracker = CardArchetypeTracker(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            if agent == eval_agent:
                tracker.before_decision(env)

            action = role_action(env, agent, obs_dict, roles, policies, bot)
            event = decode_planned_event(env, agent, action)
            pre_eval_hand = list(env._hands.get(eval_agent, []))
            known_top = env._deck_knowledge.get(eval_agent, {}).get(0)

            if agent == eval_agent:
                tracker.after_action(event)
                tactical.before_eval_action(env, event)

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tactical.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tactical.finish_game(env, bool(won)))
        card_summary = tracker.finish()
        records.append(
            {
                "seed": seed,
                "config": config_name,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "elimination_order": elimination_order,
                "roles": roles,
                "clusters": card_summary["clusters"],
                "phase_decisions": card_summary["phase_decisions"],
                "card_phase_decisions": card_summary["card_phase_decisions"],
                "group_phase_decisions": card_summary["group_phase_decisions"],
                "played_card_phase": card_summary["played_card_phase"],
                "played_card_total": card_summary["played_card_total"],
                "flags": card_summary["flags"],
                "_card_phase_game_keys": card_summary["_card_phase_game_keys"],
                "_group_phase_game_keys": card_summary["_group_phase_game_keys"],
            }
        )

    summary = summarize_records(records)
    summary["composite_component_label"] = CONFIG_LABELS[config_name]
    summary["tactical_raw_counts"] = merge_counter_dict(aggregate_tactical)
    logger.write(
        f"Step3 fast clustering termine {config_name}",
        expected="Collecter archetypes, winrate, et positions de defaite.",
        actual=(
            f"games={games}, winrate={summary['winrate']:.4f}, "
            f"first_out_loss={summary['loss_positions']['first_out']['rate_among_losses']:.4f}"
        ),
    )
    return {"summary": summary, "records": records}


def records_with_cluster(records: list[dict], cluster: str) -> list[dict]:
    return [record for record in records if cluster in set(record["clusters"])]


def records_with_key(records: list[dict], field: str, key: str) -> list[dict]:
    return [record for record in records if int(record.get(field, {}).get(key, 0)) > 0]


def summarize_clusters(records: list[dict]) -> dict:
    result = {}
    for cluster, label in CLUSTER_LABELS.items():
        subset = records_with_cluster(records, cluster)
        if not subset:
            continue
        result[cluster] = {
            "label": label,
            "summary": summarize_records(subset),
        }
    return result


def summarize_group_phase(records: list[dict]) -> dict:
    result = {}
    for group in CARD_GROUPS:
        result[group] = {
            "label": CARD_GROUP_LABELS[group],
            "cards": sorted_card_names(CARD_GROUPS[group]),
            "phases": {},
        }
        for phase in PHASES:
            key = f"{group}|{phase}"
            subset = records_with_key(records, "group_phase_decisions", key)
            decisions = sum(int(record.get("group_phase_decisions", {}).get(key, 0)) for record in records)
            result[group]["phases"][phase] = {
                "games_with_group": int(len(subset)),
                "game_rate": float(len(subset) / max(1, len(records))),
                "decision_exposures": int(decisions),
                "summary": summarize_records(subset) if subset else None,
            }
    return result


def summarize_card_phase(records: list[dict]) -> dict:
    result = {}
    for card in range(10):
        card_name = CARD_NAMES[card]
        result[card_name] = {}
        for phase in PHASES:
            key = f"{card_name}|{phase}"
            subset = records_with_key(records, "card_phase_decisions", key)
            decisions = sum(int(record.get("card_phase_decisions", {}).get(key, 0)) for record in records)
            plays = sum(int(record.get("played_card_phase", {}).get(key, 0)) for record in records)
            result[card_name][phase] = {
                "games_with_card": int(len(subset)),
                "game_rate": float(len(subset) / max(1, len(records))),
                "decision_exposures": int(decisions),
                "plays": int(plays),
                "summary": summarize_records(subset) if subset else None,
            }
    return result


def aggregate_configs(configs: dict) -> dict:
    configs_for_composite = {
        name: {"winrate": value["summary"]["winrate"]}
        for name, value in configs.items()
    }
    all_records = []
    for config in configs.values():
        all_records.extend(config["records"])
    return {
        "overall": summarize_records(all_records),
        "by_config": {name: value["summary"] for name, value in configs.items()},
        "composite": composite_score(configs_for_composite),
        "clusters": summarize_clusters(all_records),
        "group_phase": summarize_group_phase(all_records),
        "card_phase": summarize_card_phase(all_records),
    }


def top_low_win_clusters(clusters: dict, min_games: int) -> list[tuple[str, dict]]:
    rows = []
    for name, payload in clusters.items():
        summary = payload["summary"]
        if summary["games"] >= min_games:
            rows.append((name, payload))
    rows.sort(key=lambda item: (item[1]["summary"]["winrate"], -item[1]["summary"]["games"]))
    return rows[:8]


def write_markdown(payload: dict, path: Path) -> None:
    result = payload["results"]
    lines = [
        "# Step4 - Analyse Des Faiblesses Par Cartes",
        "",
        f"Date: {payload['created_at']}.",
        "",
        "Modele analyse: `Step3 rapide DAgger`.",
        "",
        f"Checkpoint: `{payload['checkpoint']}`.",
        "",
        "Taxonomie: `step4_weakness_analysis/CARD_TAXONOMY.md`.",
        "",
        "## Resultat Global",
        "",
        "| Games | Composite | Winrate moyen | Reward moyen |",
        "|---:|---:|---:|---:|",
        (
            f"| {result['overall']['games']} | {result['composite']:.5f} | "
            f"{pct(result['overall']['winrate'])} | {result['overall']['mean_reward']:.4f} |"
        ),
        "",
        "## Arena Fair Seat-Rotated",
        "",
        "| Composition | Games | Winrate | Reward moyen | 1er sorti parmi pertes | 2e | 3e | Finaliste perdant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name in CONFIG_HEURISTIC_COUNTS:
        row = result["by_config"][config_name]
        loss = row["loss_positions"]
        lines.append(
            "| {label} | {games} | {wr} | {reward:.4f} | {first} | {second} | {third} | {final} |".format(
                label=CONFIG_LABELS[config_name],
                games=row["games"],
                wr=pct(row["winrate"]),
                reward=row["mean_reward"],
                first=pct(loss["first_out"]["rate_among_losses"]),
                second=pct(loss["second_out"]["rate_among_losses"]),
                third=pct(loss["third_out"]["rate_among_losses"]),
                final=pct(loss["final_loser"]["rate_among_losses"]),
            )
        )

    overall_loss = result["overall"]["loss_positions"]
    lines.extend(
        [
            "",
            "## Positions De Defaite Globales",
            "",
            "Les pourcentages ci-dessous sont conditionnels aux defaites: leur somme vaut 100%.",
            "",
            "| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |",
            "|---:|---:|---:|---:|",
            (
                f"| {pct(overall_loss['first_out']['rate_among_losses'])} | "
                f"{pct(overall_loss['second_out']['rate_among_losses'])} | "
                f"{pct(overall_loss['third_out']['rate_among_losses'])} | "
                f"{pct(overall_loss['final_loser']['rate_among_losses'])} |"
            ),
            "",
            "## Archetypes De Parties",
            "",
            "| Archetype | Games | Winrate | 1er sorti / pertes | 2e | 3e | Finaliste perdant |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cluster, payload_cluster in sorted(
        result["clusters"].items(),
        key=lambda item: (-item[1]["summary"]["games"], item[0]),
    ):
        row = payload_cluster["summary"]
        loss = row["loss_positions"]
        lines.append(
            "| {label} | {games} | {wr} | {first} | {second} | {third} | {final} |".format(
                label=payload_cluster["label"],
                games=row["games"],
                wr=pct(row["winrate"]),
                first=pct(loss["first_out"]["rate_among_losses"]),
                second=pct(loss["second_out"]["rate_among_losses"]),
                third=pct(loss["third_out"]["rate_among_losses"]),
                final=pct(loss["final_loser"]["rate_among_losses"]),
            )
        )

    min_games = max(100, int(result["overall"]["games"] * 0.03))
    weak_clusters = top_low_win_clusters(result["clusters"], min_games)
    lines.extend(
        [
            "",
            "## Archetypes A Surveiller",
            "",
            f"Filtre: au moins {min_games} parties.",
            "",
            "| Archetype | Games | Winrate | Lecture rapide |",
            "|---|---:|---:|---|",
        ]
    )
    for cluster, payload_cluster in weak_clusters:
        summary = payload_cluster["summary"]
        lines.append(
            f"| {payload_cluster['label']} | {summary['games']} | {pct(summary['winrate'])} | "
            "Candidat de faiblesse si confirme par seed independant. |"
        )

    lines.extend(
        [
            "",
            "## Familles De Cartes Par Phase",
            "",
            "Chaque case indique: `presence dans les parties / winrate de ces parties`.",
            "",
            "| Famille | Cartes | Early | Mid | Late |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for group, payload_group in result["group_phase"].items():
        cells = []
        for phase in PHASES:
            phase_payload = payload_group["phases"][phase]
            summary = phase_payload["summary"]
            wr = pct(summary["winrate"]) if summary else "n/a"
            cells.append(f"{pct(phase_payload['game_rate'])} / {wr}")
        lines.append(
            f"| {payload_group['label']} | {payload_group['cards']} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )

    lines.extend(
        [
            "",
            "## Cartes Individuelles Par Phase",
            "",
            "Chaque case indique: `presence dans les parties / winrate / coups joues`.",
            "",
            "| Carte | Early | Mid | Late |",
            "|---|---:|---:|---:|",
        ]
    )
    for card_name, phase_payloads in result["card_phase"].items():
        cells = []
        for phase in PHASES:
            phase_payload = phase_payloads[phase]
            summary = phase_payload["summary"]
            wr = pct(summary["winrate"]) if summary else "n/a"
            cells.append(f"{pct(phase_payload['game_rate'])} / {wr} / {phase_payload['plays']}")
        lines.append(f"| {card_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Les clusters sont multi-label: une meme partie peut etre `Princesse tot` et `Controle main/pioche`.",
            "- Les phases sont basees sur la taille de pioche au moment ou le modele agit.",
            "- Ce rapport sert a trouver des hypotheses de faiblesse, pas a conclure seul. Les plus gros signaux devront etre retestes avec un seed independant.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Step4 card/phase clustering for the latest Step3 fast model.")
    parser.add_argument("--games", type=int, default=1000, help="Games per composition.")
    parser.add_argument("--seed-start", type=int, default=310000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="step3_fast_card_clusters_1000.json")
    parser.add_argument("--markdown", default="step3_fast_card_clusters_1000.md")
    parser.add_argument("--run-log", default="step4_weakness_analysis/logs/2026-04-25_step3_fast_card_clusters_1000.md")
    parser.add_argument("--print-json", action="store_true", help="Print the full JSON payload to stdout.")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step4 clustering cartes",
        expected="Identifier les types de parties ou Step3 rapide gagne/perd.",
        actual=f"games={args.games} par composition, checkpoint={args.step3_fast_checkpoint}",
        details=vars(args),
    )

    context_args = SimpleNamespace(**vars(args))
    context = prepare_policy_context("step3_fast_dagger", context_args)
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"Debut config {config_name}",
            expected="Seat rotation + heuristiques fair shuffle_targets=True.",
            actual=f"seed_start={seed_start}",
        )
        configs[config_name] = evaluate_config(config_name, args.games, seed_start, context_args, context, logger)

    results = aggregate_configs(configs)
    payload = {
        "created_at": now_stamp(),
        "checkpoint": str(context["checkpoint"]),
        "base_checkpoint": str(context["base_checkpoint"]),
        "games_per_config": args.games,
        "seed_start": args.seed_start,
        "taxonomy": {
            "card_groups": {
                key: {
                    "label": CARD_GROUP_LABELS[key],
                    "cards": {CARD_NAMES[card]: card for card in sorted(cards)},
                }
                for key, cards in CARD_GROUPS.items()
            },
            "phases": {
                "early": "deck_remaining >= 11",
                "mid": "6 <= deck_remaining <= 10",
                "late": "deck_remaining <= 5",
            },
        },
        "results": results,
    }
    clean_payload = json_safe(payload)
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(clean_payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_markdown(clean_payload, markdown)
    logger.write(
        "Fin Step4 clustering cartes",
        expected="Produire JSON + Markdown pour audit des faiblesses.",
        actual=f"json={output}, markdown={markdown}, composite={results['composite']:.5f}",
    )
    if args.print_json:
        print(json.dumps(clean_payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(
            json.dumps(
                {
                    "created_at": clean_payload["created_at"],
                    "games_per_config": clean_payload["games_per_config"],
                    "composite": results["composite"],
                    "overall_winrate": results["overall"]["winrate"],
                    "json": str(output),
                    "markdown": str(markdown),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
