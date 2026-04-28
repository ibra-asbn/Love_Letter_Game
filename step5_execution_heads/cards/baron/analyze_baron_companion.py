"""Audit Baron usage by companion card for the current Step3 policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import (
    CONFIG_HEURISTIC_COUNTS,
    CONFIG_LABELS,
    build_roles,
    decode_planned_event,
    make_policy,
    prepare_policy_context,
    role_action,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import now_stamp
from step2_rl_finetune.evaluate_step2 import random_action


CARD_NAMES = {
    0: "Espionne",
    1: "Garde",
    2: "Pretre",
    3: "Baron",
    4: "Servante",
    5: "Prince",
    6: "Chancelier",
    7: "Roi",
    8: "Comtesse",
    9: "Princesse",
}


def pct(num: int | float, den: int | float) -> str:
    return f"{100.0 * float(num) / max(1.0, float(den)):.2f}%"


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def action_card(action: int) -> int:
    return int(action) // 100


def companion_for_baron(hand: list[int]) -> int | None:
    if 3 not in hand:
        return None
    others = [int(card) for card in hand if int(card) != 3]
    if others:
        return int(others[0])
    return 3


def phase_for_deck(deck_len: int) -> str:
    if deck_len >= 11:
        return "early"
    if deck_len >= 6:
        return "mid"
    return "late"


def summarize(counter: Counter) -> dict:
    opportunities = counter["opportunities"]
    played = counter["played"]
    return {
        "opportunities": int(opportunities),
        "played": int(played),
        "play_rate": float(played / max(1, opportunities)),
        "forced": int(counter["forced"]),
        "forced_rate": float(counter["forced"] / max(1, opportunities)),
        "played_winrate": float(counter["played_wins"] / max(1, played)),
        "played_mean_reward": float(counter["played_reward_sum"] / max(1, played)),
        "opportunity_winrate": float(counter["opportunity_wins"] / max(1, opportunities)),
        "duel_win_rate": float(counter["duel_win"] / max(1, played)),
        "duel_loss_rate": float(counter["duel_loss"] / max(1, played)),
        "duel_tie_rate": float(counter["duel_tie"] / max(1, played)),
        "duel_no_target_rate": float(counter["duel_no_target"] / max(1, played)),
    }


def make_markdown(payload: dict) -> str:
    lines = [
        "# Audit Baron par carte accompagnee",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Modele audite: `{payload['policy_name']}`.",
        f"Parties: `{payload['games']}` par composition d'arene.",
        "",
        "Lecture: `Pct joue Baron` = parmi les tours ou le modele a `Baron + carte`, "
        "pourcentage ou il choisit de jouer Baron. `Winrate si joue` = victoire finale "
        "de la manche apres ces actions Baron. Les duels sont mesures immediatement.",
        "",
        "## Global",
        "",
        "| Carte avec Baron | Opportunites | Pct joue Baron | Force | Winrate si joue | Reward si joue | Duel gagne | Duel perdu | Egalite | Sans cible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for card, row in payload["global_rows"]:
        label = f"{card} {CARD_NAMES[int(card)]}"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["opportunities"]),
                    f"{100.0 * row['play_rate']:.2f}%",
                    f"{100.0 * row['forced_rate']:.2f}%",
                    f"{100.0 * row['played_winrate']:.2f}%",
                    f"{row['played_mean_reward']:.3f}",
                    f"{100.0 * row['duel_win_rate']:.2f}%",
                    f"{100.0 * row['duel_loss_rate']:.2f}%",
                    f"{100.0 * row['duel_tie_rate']:.2f}%",
                    f"{100.0 * row['duel_no_target_rate']:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Par Composition", ""])
    for config_name, rows in payload["config_rows"].items():
        lines.extend(
            [
                f"### {CONFIG_LABELS.get(config_name, config_name)}",
                "",
                "| Carte avec Baron | Opportunites | Pct joue Baron | Winrate si joue | Duel gagne | Duel perdu |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for card, row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{card} {CARD_NAMES[int(card)]}",
                        str(row["opportunities"]),
                        f"{100.0 * row['play_rate']:.2f}%",
                        f"{100.0 * row['played_winrate']:.2f}%",
                        f"{100.0 * row['duel_win_rate']:.2f}%",
                        f"{100.0 * row['duel_loss_rate']:.2f}%",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Step3 Baron usage by companion card.")
    parser.add_argument("--policy-name", default="step3_fast_dagger")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=1800000)
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
    parser.add_argument("--output-json", default="step5_execution_heads/cards/baron/reports/baron_companion_audit_step3.json")
    parser.add_argument("--output-md", default="step5_execution_heads/cards/baron/reports/baron_companion_audit_step3.md")
    args = parser.parse_args()

    context = prepare_policy_context(args.policy_name, args)
    global_counts = defaultdict(Counter)
    config_counts = defaultdict(lambda: defaultdict(Counter))
    phase_counts = defaultdict(lambda: defaultdict(Counter))

    for config_index, (config_name, heuristic_count) in enumerate(CONFIG_HEURISTIC_COUNTS.items()):
        env = LoveLetterRLEnv(num_players=4)
        bot = HeuristicBot(shuffle_targets=True)
        seed_start = args.seed_start + config_index * args.seed_stride
        for game in range(args.games):
            seed = seed_start + game
            np.random.seed(seed)
            env.reset(seed=seed)
            eval_agent = f"player_{game % 4}"
            roles = build_roles(eval_agent, heuristic_count, game)
            policies = {eval_agent: make_policy(args.policy_name, args, roles, eval_agent, context)}
            game_events = []
            rewards = {agent: 0.0 for agent in env.possible_agents}

            for _turn, agent in enumerate(env.agent_iter()):
                obs_dict, reward, terminated, truncated, _info = env.last()
                rewards[agent] += float(reward)
                if terminated or truncated:
                    env.step(None)
                    continue

                action = role_action(env, agent, obs_dict, roles, policies, bot)

                if agent == eval_agent and not env._chancellor_pending:
                    hand = [int(card) for card in env._hands.get(agent, [])]
                    companion = companion_for_baron(hand)
                    if companion is not None:
                        mask = obs_dict["action_mask"]
                        legal_cards = {action_card(item) for item in valid_actions(mask)}
                        event = decode_planned_event(env, agent, action)
                        played_baron = bool(event["kind"] == "card" and int(event["card"]) == 3)
                        target_card = event.get("target_card")
                        remaining = event.get("remaining_hand", [])
                        kept = int(remaining[0]) if remaining else companion
                        duel = "not_played"
                        if played_baron:
                            if target_card is None:
                                duel = "duel_no_target"
                            elif kept > int(target_card):
                                duel = "duel_win"
                            elif int(target_card) > kept:
                                duel = "duel_loss"
                            else:
                                duel = "duel_tie"
                        game_events.append(
                            {
                                "config": config_name,
                                "companion": companion,
                                "played": played_baron,
                                "forced": legal_cards == {3},
                                "duel": duel,
                                "phase": phase_for_deck(len(env._deck)),
                            }
                        )
                env.step(action)

            reward_eval = float(rewards[eval_agent])
            won = int(reward_eval >= 1.0)
            for event in game_events:
                companion = event["companion"]
                counters = [
                    global_counts[companion],
                    config_counts[event["config"]][companion],
                    phase_counts[event["phase"]][companion],
                ]
                for counter in counters:
                    counter["opportunities"] += 1
                    counter["opportunity_wins"] += won
                    if event["forced"]:
                        counter["forced"] += 1
                    if event["played"]:
                        counter["played"] += 1
                        counter["played_wins"] += won
                        counter["played_reward_sum"] += reward_eval
                        counter[event["duel"]] += 1

    global_rows = [(card, summarize(global_counts[card])) for card in sorted(global_counts)]
    config_rows = {
        config: [(card, summarize(rows[card])) for card in sorted(rows)]
        for config, rows in config_counts.items()
    }
    phase_rows = {
        phase: [(card, summarize(rows[card])) for card in sorted(rows)]
        for phase, rows in phase_counts.items()
    }
    payload = {
        "created_at": now_stamp(),
        "policy_name": args.policy_name,
        "games": args.games,
        "args": vars(args),
        "global_rows": global_rows,
        "config_rows": config_rows,
        "phase_rows": phase_rows,
    }
    output_json = PROJECT_ROOT / args.output_json
    output_md = PROJECT_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "global_rows": global_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
