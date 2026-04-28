"""Rank cards by potential gain from rollout-guided action-value search.

This is an analysis probe, not a training script. It samples player_0 turns
from the Step2 policy, evaluates candidate actions by determinized rollouts,
and aggregates where the model seems to leave winrate on the table.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import ModelSeat, OPPONENT_CONFIGS, random_action
from step3_action_value.evaluate_rollout_guided import evaluate_candidate_actions
from step3_action_value.mini_rollout_probe import CARD_NAMES, choose_actions_for_probe, decode_action


STEP_DIR = PROJECT_ROOT / "step3_action_value"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def action_card(action: int) -> int | None:
    if 900 <= int(action) <= 905:
        return None
    return int(action) // 100


def opponent_action(env, agent, obs_dict, mode, bot, opponents):
    if mode == "opponent_config":
        mode = opponents[agent]
    if mode == "heuristic":
        return bot.choose_action(env, agent)
    if mode == "random":
        return random_action(obs_dict)
    if mode == "mixed":
        return bot.choose_action(env, agent) if np.random.rand() < 0.5 else random_action(obs_dict)
    raise ValueError(mode)


def candidate_actions(env, model_action, heuristic_action, max_actions):
    actions = choose_actions_for_probe(env, max_actions)
    for forced in [int(model_action), int(heuristic_action)]:
        if forced not in actions:
            actions = [forced] + actions
    return list(dict.fromkeys(actions))[:max_actions]


def best_rows_by_card(rows):
    best = {}
    for row in rows:
        card = action_card(row["action"])
        if card is None:
            continue
        current = best.get(card)
        if current is None or (row["winrate"], row["mean_reward"]) > (
            current["winrate"],
            current["mean_reward"],
        ):
            best[card] = row
    return best


def compact_state(env, model_action, heuristic_action):
    hand = list(env._hands.get("player_0", []))
    known = {}
    for opp in env.possible_agents:
        if opp == "player_0":
            continue
        idx = np.where(env._known_cards["player_0"][opp] >= 1.0)[0]
        if len(idx):
            known[opp] = CARD_NAMES[int(idx[0])]
    return {
        "hand": [CARD_NAMES[c] for c in hand],
        "hand_ids": [int(c) for c in hand],
        "deck_size": len(env._deck),
        "protected": {a: bool(env._protected.get(a, False)) for a in env.possible_agents},
        "known_cards": known,
        "played_counts": {
            a: [CARD_NAMES[c] for c in env._played_cards.get(a, [])]
            for a in env.possible_agents
        },
        "model_action": int(model_action),
        "model_decoded": decode_action(int(model_action)),
        "heuristic_action": int(heuristic_action),
        "heuristic_decoded": decode_action(int(heuristic_action)),
    }


def collect_probe_states(args, checkpoint, opponents, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    per_card_counts = Counter()
    total_player0_decisions = 0

    for game in range(args.collect_games):
        if all(per_card_counts[c] >= args.states_per_card for c in args.cards):
            break

        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        model = ModelSeat(checkpoint)

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                model_action = int(model.act(obs_dict, agent))
                total_player0_decisions += 1

                hand = sorted(set(int(c) for c in env._hands.get(agent, [])))
                tracked_cards = [
                    c for c in hand
                    if c in args.cards and per_card_counts[c] < args.states_per_card
                ]

                if (
                    tracked_cards
                    and not env._chancellor_pending
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    heuristic_action = int(bot.choose_action(env, agent))
                    actions = candidate_actions(env, model_action, heuristic_action, args.max_actions)
                    rows = evaluate_candidate_actions(
                        env,
                        actions,
                        checkpoint,
                        opponents,
                        args,
                        decision_seed=seed * 100 + turn,
                    )
                    by_action = {row["action"]: row for row in rows}
                    by_card = best_rows_by_card(rows)
                    model_row = by_action.get(model_action)
                    if model_row is not None and rows:
                        best = rows[0]
                        model_card = action_card(model_action)
                        model_card_best = by_card.get(model_card, model_row)
                        best_other_card = None
                        for row in rows:
                            if action_card(row["action"]) != model_card:
                                best_other_card = row
                                break
                        execution_regret = max(
                            0.0,
                            float(model_card_best["winrate"] - model_row["winrate"]),
                        )
                        card_selection_regret = (
                            max(0.0, float(best_other_card["winrate"] - model_card_best["winrate"]))
                            if best_other_card is not None
                            else 0.0
                        )
                        record = {
                            "seed": seed,
                            "turn": turn,
                            "tracked_cards": tracked_cards,
                            "state": compact_state(env, model_action, heuristic_action),
                            "model_action": int(model_action),
                            "model_card": model_card,
                            "best_action": int(best["action"]),
                            "best_card": action_card(best["action"]),
                            "model_card_best_action": int(model_card_best["action"]),
                            "model_card_best_winrate": float(model_card_best["winrate"]),
                            "model_card_best_mean_reward": float(model_card_best["mean_reward"]),
                            "best_other_card_action": int(best_other_card["action"]) if best_other_card else None,
                            "best_other_card": action_card(best_other_card["action"]) if best_other_card else None,
                            "best_other_card_winrate": float(best_other_card["winrate"]) if best_other_card else None,
                            "execution_regret_winrate": execution_regret,
                            "card_selection_regret_winrate": card_selection_regret,
                            "card_values": {
                                str(card): {
                                    "action": int(row["action"]),
                                    "decoded": row["decoded"],
                                    "winrate": float(row["winrate"]),
                                    "mean_reward": float(row["mean_reward"]),
                                }
                                for card, row in by_card.items()
                            },
                            "model_winrate": float(model_row["winrate"]),
                            "best_winrate": float(best["winrate"]),
                            "model_mean_reward": float(model_row["mean_reward"]),
                            "best_mean_reward": float(best["mean_reward"]),
                            "regret_winrate": float(best["winrate"] - model_row["winrate"]),
                            "regret_reward": float(best["mean_reward"] - model_row["mean_reward"]),
                            "top_actions": rows[: min(8, len(rows))],
                        }
                        records.append(record)
                        for card in tracked_cards:
                            per_card_counts[card] += 1

                        if len(records) % args.log_every_states == 0:
                            logger.write(
                                "Collecte probe potentiel par carte",
                                expected="Equilibrer les exemples entre cartes jouables.",
                                actual=f"states={len(records)}, counts={dict(per_card_counts)}",
                            )

                env.step(model_action)
            else:
                action = opponent_action(env, agent, obs_dict, args.collect_opponents, bot, opponents)
                env.step(action)

    return records, per_card_counts, total_player0_decisions


def attributed_for_card(card, record, margin):
    card_value = record["card_values"].get(str(card))
    model_card = record["model_card"]
    if card_value is None or model_card is None:
        return {
            "potential": 0.0,
            "execution": 0.0,
            "missed": 0.0,
            "avoid": 0.0,
            "kind": "unavailable",
            "high_margin": False,
        }

    if card == model_card:
        execution = float(record["execution_regret_winrate"])
        avoid = float(record["card_selection_regret_winrate"])
        potential = execution + avoid
        if avoid >= margin:
            kind = "avoid_card"
        elif execution >= margin:
            kind = "refine_same_card"
        else:
            kind = "kept"
        return {
            "potential": potential,
            "execution": execution,
            "missed": 0.0,
            "avoid": avoid,
            "kind": kind,
            "high_margin": potential >= margin,
        }

    # Compare a different card against the best possible version of the card the
    # model chose. This avoids blaming Comtesse for a badly targeted Guard, etc.
    missed = max(0.0, float(card_value["winrate"] - record["model_card_best_winrate"]))
    return {
        "potential": missed,
        "execution": 0.0,
        "missed": missed,
        "avoid": 0.0,
        "kind": "missed_card" if missed >= margin else "not_better_than_chosen_card",
        "high_margin": missed >= margin,
    }


def summarize_card(card, records_for_card, total_player0_decisions, margin):
    opportunities = len(records_for_card)
    if opportunities == 0:
        return {
            "opportunities": 0,
            "mean_regret": 0.0,
            "mean_attributed_potential": 0.0,
            "expected_gain_per_player0_decision": 0.0,
            "high_margin_rate": 0.0,
        }

    regrets = np.array([r["regret_winrate"] for r in records_for_card], dtype=np.float64)
    attributions = [attributed_for_card(card, r, margin) for r in records_for_card]
    potentials = np.array([a["potential"] for a in attributions], dtype=np.float64)
    executions = np.array([a["execution"] for a in attributions], dtype=np.float64)
    missed_values = np.array([a["missed"] for a in attributions], dtype=np.float64)
    avoid_values = np.array([a["avoid"] for a in attributions], dtype=np.float64)
    high = potentials >= margin
    model_plays = [r for r in records_for_card if r["model_card"] == card]
    best_plays = [r for r in records_for_card if r["best_card"] == card]
    missed = [a for a in attributions if a["kind"] == "missed_card"]
    avoid = [a for a in attributions if a["kind"] == "avoid_card"]
    refine = [a for a in attributions if a["kind"] == "refine_same_card"]

    return {
        "opportunities": int(opportunities),
        "mean_attributed_potential": float(potentials.mean()),
        "median_attributed_potential": float(np.median(potentials)),
        "max_attributed_potential": float(potentials.max()),
        "sum_attributed_potential": float(potentials.sum()),
        "expected_attributed_gain_per_player0_decision": float(
            potentials.sum() / max(1, total_player0_decisions)
        ),
        "mean_execution_regret": float(executions.mean()),
        "mean_missed_card_gain": float(missed_values.mean()),
        "mean_avoid_card_gain": float(avoid_values.mean()),
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(regrets.max()),
        "sum_regret": float(regrets.sum()),
        "expected_gain_per_player0_decision": float(regrets.sum() / max(1, total_player0_decisions)),
        "high_margin_count": int(high.sum()),
        "high_margin_rate": float(high.mean()),
        "model_play_rate_when_seen": float(len(model_plays) / opportunities),
        "best_play_rate_when_seen": float(len(best_plays) / opportunities),
        "missed_card_count": int(len(missed)),
        "missed_card_rate": float(len(missed) / opportunities),
        "avoid_card_count": int(len(avoid)),
        "avoid_card_rate": float(len(avoid) / opportunities),
        "refine_same_card_count": int(len(refine)),
        "refine_same_card_rate": float(len(refine) / opportunities),
    }


def summarize(records, total_player0_decisions, args):
    by_card_records = {card: [] for card in args.cards}
    for record in records:
        for card in record["tracked_cards"]:
            by_card_records[card].append(record)

    cards = {}
    for card in args.cards:
        cards[str(card)] = {
            "card": card,
            "card_name": CARD_NAMES[card],
            **summarize_card(card, by_card_records[card], total_player0_decisions, args.high_margin),
        }

    ranked_by_expected_gain = sorted(
        cards.values(),
        key=lambda row: row["expected_attributed_gain_per_player0_decision"],
        reverse=True,
    )
    ranked_by_mean_regret = sorted(
        cards.values(),
        key=lambda row: row["mean_attributed_potential"],
        reverse=True,
    )

    examples = {}
    for card, rows in by_card_records.items():
        sorted_rows = sorted(rows, key=lambda r: r["regret_winrate"], reverse=True)
        examples[str(card)] = [
            {
                "card": CARD_NAMES[card],
                "regret_winrate": row["regret_winrate"],
                "attributed": attributed_for_card(card, row, args.high_margin),
                "model_card_best": decode_action(row["model_card_best_action"])["label"],
                "regret_reward": row["regret_reward"],
                "hand": row["state"]["hand"],
                "model": row["state"]["model_decoded"]["label"],
                "best": decode_action(row["best_action"])["label"],
                "top_actions": row["top_actions"][:5],
            }
            for row in sorted_rows[: args.examples_per_card]
        ]

    return {
        "cards": cards,
        "ranked_by_expected_gain": ranked_by_expected_gain,
        "ranked_by_mean_regret": ranked_by_mean_regret,
        "examples": examples,
    }


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Probe rollout-guided potential by card.")
    parser.add_argument("--checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--opponent-config", choices=sorted(OPPONENT_CONFIGS), default="vs_3H")
    parser.add_argument("--collect-opponents", choices=["opponent_config", "heuristic", "random", "mixed"], default="opponent_config")
    parser.add_argument("--collect-games", type=int, default=2500)
    parser.add_argument("--states-per-card", type=int, default=14)
    parser.add_argument("--cards", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--high-margin", type=float, default=0.12)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--example-limit", type=int, default=0)
    parser.add_argument("--examples-per-card", type=int, default=3)
    parser.add_argument("--output", default="step3_card_potential_probe.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_card_potential_probe.md")
    parser.add_argument("--log-every-states", type=int, default=25)
    parser.add_argument("--seed", type=int, default=9600)
    args = parser.parse_args()

    args.cards = sorted(set(args.cards))
    checkpoint = resolve_checkpoint(args.checkpoint)
    opponents = OPPONENT_CONFIGS[args.opponent_config]
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output

    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut probe potentiel par carte",
        expected=(
            "Identifier les cartes ou le search action-value a le plus de marge "
            "par rapport a Step2."
        ),
        actual=f"checkpoint={checkpoint}, opponent_config={args.opponent_config}",
        details=vars(args),
    )

    records, counts, total_player0_decisions = collect_probe_states(args, checkpoint, opponents, logger)
    summary = summarize(records, total_player0_decisions, args)
    report = {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "args": vars(args),
        "collection": {
            "states": len(records),
            "per_card_counts": {CARD_NAMES[k]: int(v) for k, v in counts.items()},
            "total_player0_decisions_scanned": int(total_player0_decisions),
        },
        **summary,
        "limitations": [
            "Probe exploratoire: echantillon volontairement petit.",
            "Les rollouts sont determinises et ne remplacent pas une vraie resolution du POMDP.",
            "Les actions candidates sont limitees par max-actions, donc certains coups peuvent manquer.",
            "Les cartes en main partagent parfois le meme regret d'etat; lire missed/avoid/refine pour affiner.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin probe potentiel par carte",
        expected="Obtenir un ranking des cartes a prioriser pour Step3.",
        actual=(
            "top_expected="
            + ", ".join(
                f"{row['card_name']}:{row['expected_attributed_gain_per_player0_decision']:.4f}"
                for row in report["ranked_by_expected_gain"][:5]
            )
        ),
        details={
            "collection": report["collection"],
            "ranked_by_expected_gain": report["ranked_by_expected_gain"],
            "ranked_by_mean_regret": report["ranked_by_mean_regret"],
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
