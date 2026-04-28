"""Audit how Priest information is used later by the evaluated policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import (
    CONFIG_HEURISTIC_COUNTS,
    CONFIG_LABELS,
    build_roles,
    classify_outcome,
    decode_planned_event,
    make_policy,
    prepare_policy_context,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step2_rl_finetune.evaluate_step2 import random_action
from step5_execution_heads.cards.baron.evaluate_baron_specialist import direct_eliminations_from_event
from step5_execution_heads.target_head import action_card, action_target


REPORT_DIR = PROJECT_ROOT / "step5_execution_heads/cards/priest/reports"
LOG_DIR = PROJECT_ROOT / "step5_execution_heads/cards/priest/logs"

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

EXPLOIT_CARDS = {
    1: "guard",
    3: "baron",
    5: "prince",
    7: "king",
}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def phase_name(deck_remaining: int) -> str:
    if deck_remaining >= 11:
        return "early"
    if deck_remaining >= 6:
        return "mid"
    return "late"


def role_action(env, agent: str, obs_dict, roles: dict[str, str], policies: dict[str, object], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "model":
        return int(policies[agent].act(env, obs_dict, agent))
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


def invalidate_spot(spots: list[dict], active: dict[str, int], target: str, reason: str, turn: int) -> None:
    spot_id = active.pop(target, None)
    if spot_id is None:
        return
    spots[spot_id]["active_end_turn"] = int(turn)
    spots[spot_id]["end_reason"] = reason


def mark_exploit(spots: list[dict], active: dict[str, int], event: dict, turn: int) -> None:
    target = event.get("target")
    if target not in active:
        return
    card = int(event["card"])
    method = EXPLOIT_CARDS.get(card)
    if method is None:
        return
    spot = spots[active[target]]
    seen = int(spot["seen_card"])
    target_card = event.get("target_card")
    exploit = {
        "turn": int(turn),
        "method": method,
        "action_card": card,
        "guess": event.get("guess"),
        "target_card": target_card,
        "seen_card": seen,
    }
    if method == "guard":
        exploit["uses_seen_card"] = bool(event.get("guess") == seen)
        exploit["success"] = bool(target_card == event.get("guess"))
        if not exploit["uses_seen_card"]:
            spot["off_card_guard_attempts"] += 1
            return
    elif method == "baron":
        my_after = event.get("remaining_hand", [])
        my_val = int(my_after[0]) if my_after else None
        exploit["my_card"] = my_val
        exploit["result"] = "unknown"
        if my_val is not None and target_card is not None:
            if my_val > int(target_card):
                exploit["result"] = "win"
            elif int(target_card) > my_val:
                exploit["result"] = "loss"
            else:
                exploit["result"] = "tie"
    elif method == "prince":
        exploit["princess_hit"] = bool(target_card == 9)
    elif method == "king":
        my_after = event.get("remaining_hand", [])
        exploit["my_old_remaining_card"] = int(my_after[0]) if my_after else None
    spot["exploits"].append(exploit)
    spot[f"used_{method}"] = True
    spot["used_any"] = True


def update_active_spots_after_action(
    spots: list[dict],
    active: dict[str, int],
    env,
    agent: str,
    event: dict,
    direct_eliminated: list[str],
    turn: int,
) -> None:
    for eliminated in direct_eliminated:
        invalidate_spot(spots, active, eliminated, "eliminated", turn)

    if event.get("kind") != "card":
        return

    card = int(event["card"])
    target = event.get("target")

    if card == 5 and target in active:
        invalidate_spot(spots, active, target, "prince_changed_hand", turn)
    if card == 7:
        if agent in active:
            invalidate_spot(spots, active, agent, "king_swap", turn)
        if target in active:
            invalidate_spot(spots, active, target, "king_swap", turn)

    if agent in active:
        spot_id = active[agent]
        seen = int(spots[spot_id]["seen_card"])
        remaining = event.get("remaining_hand", [])
        # If the player keeps the seen card after playing a non-mutating card,
        # the Priest information is still valid. Otherwise it has gone stale.
        mutates_self = card in {5, 6, 7, 9}
        if mutates_self or seen not in [int(item) for item in remaining]:
            invalidate_spot(spots, active, agent, "target_hand_changed", turn)

    # Engine-side eliminations can also occur outside direct card effects at the
    # end of a round; keep this conservative after every step.
    for target_name in list(active):
        if env.terminations.get(target_name, False) or target_name not in env.agents:
            invalidate_spot(spots, active, target_name, "terminated", turn)


def summarize_spots(spots: list[dict], games: int, wins: int) -> dict:
    by_card = {}
    total = len(spots)
    for card in range(10):
        rows = [spot for spot in spots if int(spot["seen_card"]) == card]
        if not rows:
            by_card[str(card)] = {
                "name": CARD_NAMES[card],
                "spots": 0,
                "pct_spots": 0.0,
                "spot_winrate": 0.0,
                "any_exploit_rate": 0.0,
                "guard_rate": 0.0,
                "baron_rate": 0.0,
                "prince_rate": 0.0,
                "king_rate": 0.0,
            }
            continue
        by_card[str(card)] = {
            "name": CARD_NAMES[card],
            "spots": len(rows),
            "pct_spots": float(len(rows) / max(1, total)),
            "spot_winrate": float(sum(int(spot["won"]) for spot in rows) / max(1, len(rows))),
            "any_exploit_rate": float(sum(int(spot["used_any"]) for spot in rows) / max(1, len(rows))),
            "guard_rate": float(sum(int(spot["used_guard"]) for spot in rows) / max(1, len(rows))),
            "baron_rate": float(sum(int(spot["used_baron"]) for spot in rows) / max(1, len(rows))),
            "prince_rate": float(sum(int(spot["used_prince"]) for spot in rows) / max(1, len(rows))),
            "king_rate": float(sum(int(spot["used_king"]) for spot in rows) / max(1, len(rows))),
        }

    totals = Counter()
    for spot in spots:
        totals["used_any"] += int(spot["used_any"])
        for method in ["guard", "baron", "prince", "king"]:
            totals[f"used_{method}"] += int(spot[f"used_{method}"])
        for exploit in spot["exploits"]:
            method = exploit["method"]
            totals[f"{method}_events"] += 1
            if method == "guard" and exploit.get("success"):
                totals["guard_success"] += 1
            if method == "baron":
                totals[f"baron_{exploit.get('result', 'unknown')}"] += 1
            if method == "prince" and exploit.get("princess_hit"):
                totals["prince_princess_hits"] += 1

    return {
        "games": int(games),
        "wins": int(wins),
        "winrate": float(wins / max(1, games)),
        "spots": int(total),
        "spots_per_game": float(total / max(1, games)),
        "overall": {
            "any_exploit_rate": float(totals["used_any"] / max(1, total)),
            "guard_rate": float(totals["used_guard"] / max(1, total)),
            "baron_rate": float(totals["used_baron"] / max(1, total)),
            "prince_rate": float(totals["used_prince"] / max(1, total)),
            "king_rate": float(totals["used_king"] / max(1, total)),
            "guard_success_rate_per_guard_exploit": float(totals["guard_success"] / max(1, totals["guard_events"])),
            "baron_win_rate_per_baron_exploit": float(totals["baron_win"] / max(1, totals["baron_events"])),
            "baron_loss_rate_per_baron_exploit": float(totals["baron_loss"] / max(1, totals["baron_events"])),
            "prince_princess_hit_rate_per_prince_exploit": float(
                totals["prince_princess_hits"] / max(1, totals["prince_events"])
            ),
            "raw_counts": dict(totals),
        },
        "by_card": by_card,
    }


def evaluate_config(config_name: str, games: int, seed_start: int, args, context: dict) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    spots = []
    game_rows = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policy = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
        policies = {eval_agent: policy}
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        active = {}
        game_spot_ids = []

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            action = role_action(env, agent, obs_dict, roles, policies, bot)
            event = decode_planned_event(env, agent, action)

            if agent == eval_agent and event.get("kind") == "card":
                mark_exploit(spots, active, event, turn)
                if int(event["card"]) == 2 and event.get("target") and event.get("target_card") is not None:
                    target = event["target"]
                    invalidate_spot(spots, active, target, "replaced_by_new_priest", turn)
                    spot = {
                        "config": config_name,
                        "seed": int(seed),
                        "seat": eval_agent,
                        "target": target,
                        "target_idx": int(event["target_idx"]),
                        "seen_card": int(event["target_card"]),
                        "phase": phase_name(len(env._deck)),
                        "turn": int(turn),
                        "used_any": False,
                        "used_guard": False,
                        "used_baron": False,
                        "used_prince": False,
                        "used_king": False,
                        "off_card_guard_attempts": 0,
                        "exploits": [],
                        "active_end_turn": None,
                        "end_reason": "end_of_round",
                        "won": 0,
                        "reward": 0.0,
                        "outcome": None,
                    }
                    spots.append(spot)
                    spot_id = len(spots) - 1
                    active[target] = spot_id
                    game_spot_ids.append(spot_id)

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            update_active_spots_after_action(spots, active, env, agent, event, direct_eliminated, turn)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        outcome = classify_outcome(eval_agent, reward_eval, elimination_order)
        for spot_id in game_spot_ids:
            spots[spot_id]["won"] = won
            spots[spot_id]["reward"] = reward_eval
            spots[spot_id]["outcome"] = outcome
        game_rows.append({"seed": seed, "won": won, "reward": reward_eval, "outcome": outcome, "spots": len(game_spot_ids)})

    return {
        "config": config_name,
        "label": CONFIG_LABELS[config_name],
        "summary": summarize_spots(spots, games, sum(row["won"] for row in game_rows)),
        "spots_sample": spots[: args.example_limit],
    }


def aggregate_configs(configs: dict) -> dict:
    total_spots = []
    games = 0
    wins = 0
    by_card_acc = defaultdict(Counter)
    method_counts = Counter()
    for config in configs.values():
        summary = config["summary"]
        games += int(summary["games"])
        wins += int(summary["wins"])
        for card, row in summary["by_card"].items():
            n = int(row["spots"])
            by_card_acc[card]["spots"] += n
            by_card_acc[card]["wins"] += row["spot_winrate"] * n
            for key in ["any_exploit_rate", "guard_rate", "baron_rate", "prince_rate", "king_rate"]:
                by_card_acc[card][key] += row[key] * n
        raw = summary["overall"]["raw_counts"]
        method_counts.update(raw)
        total_spots.append(int(summary["spots"]))
    spot_total = sum(total_spots)
    by_card = {}
    for card in range(10):
        row = by_card_acc[str(card)]
        n = int(row["spots"])
        by_card[str(card)] = {
            "name": CARD_NAMES[card],
            "spots": n,
            "pct_spots": float(n / max(1, spot_total)),
            "spot_winrate": float(row["wins"] / max(1, n)),
            "any_exploit_rate": float(row["any_exploit_rate"] / max(1, n)),
            "guard_rate": float(row["guard_rate"] / max(1, n)),
            "baron_rate": float(row["baron_rate"] / max(1, n)),
            "prince_rate": float(row["prince_rate"] / max(1, n)),
            "king_rate": float(row["king_rate"] / max(1, n)),
        }
    return {
        "games": int(games),
        "wins": int(wins),
        "winrate": float(wins / max(1, games)),
        "spots": int(spot_total),
        "spots_per_game": float(spot_total / max(1, games)),
        "overall": {
            "any_exploit_rate": float(method_counts["used_any"] / max(1, spot_total)),
            "guard_rate": float(method_counts["used_guard"] / max(1, spot_total)),
            "baron_rate": float(method_counts["used_baron"] / max(1, spot_total)),
            "prince_rate": float(method_counts["used_prince"] / max(1, spot_total)),
            "king_rate": float(method_counts["used_king"] / max(1, spot_total)),
            "guard_success_rate_per_guard_exploit": float(
                method_counts["guard_success"] / max(1, method_counts["guard_events"])
            ),
            "baron_win_rate_per_baron_exploit": float(
                method_counts["baron_win"] / max(1, method_counts["baron_events"])
            ),
            "baron_loss_rate_per_baron_exploit": float(
                method_counts["baron_loss"] / max(1, method_counts["baron_events"])
            ),
            "prince_princess_hit_rate_per_prince_exploit": float(
                method_counts["prince_princess_hits"] / max(1, method_counts["prince_events"])
            ),
            "raw_counts": dict(method_counts),
        },
        "by_card": by_card,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Pretre - Audit Flux D'Information",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Policy: `{payload['policy']}`.",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "Definition: une info Pretre est consideree exploitee si le joueur cible ensuite le meme adversaire, avant invalidation de l'info, avec Garde exact, Baron, Prince ou Roi.",
        "",
        "## Synthese",
        "",
        f"- Parties totales: `{agg['games']}`.",
        f"- Winrate global: `{pct(agg['winrate'])}`.",
        f"- Spots Pretre: `{agg['spots']}` (`{agg['spots_per_game']:.3f}` par partie).",
        f"- Spots exploites au moins une fois: `{pct(agg['overall']['any_exploit_rate'])}`.",
        f"- Exploit par Garde exact: `{pct(agg['overall']['guard_rate'])}`.",
        f"- Exploit par Baron: `{pct(agg['overall']['baron_rate'])}`.",
        f"- Exploit par Prince: `{pct(agg['overall']['prince_rate'])}`.",
        f"- Exploit par Roi: `{pct(agg['overall']['king_rate'])}`.",
        "",
        "## Cartes Spottees",
        "",
        "| Carte vue | Spots | % spots | Winrate quand vue | Exploit total | Garde | Baron | Prince | Roi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for card in range(10):
        row = agg["by_card"][str(card)]
        lines.append(
            f"| {card} {row['name']} | {row['spots']} | {pct(row['pct_spots'])} | "
            f"{pct(row['spot_winrate'])} | {pct(row['any_exploit_rate'])} | "
            f"{pct(row['guard_rate'])} | {pct(row['baron_rate'])} | "
            f"{pct(row['prince_rate'])} | {pct(row['king_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Qualite Des Exploitations",
            "",
            "| Type | Qualite mesuree |",
            "|---|---:|",
            f"| Garde exact apres Pretre | {pct(agg['overall']['guard_success_rate_per_guard_exploit'])} hit |",
            f"| Baron apres Pretre | {pct(agg['overall']['baron_win_rate_per_baron_exploit'])} gagne / {pct(agg['overall']['baron_loss_rate_per_baron_exploit'])} perdu |",
            f"| Prince apres Pretre | {pct(agg['overall']['prince_princess_hit_rate_per_prince_exploit'])} Princesse touchee |",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Analyze how Priest spots are exploited later.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=3100000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--policy", default="step3_fast_dagger", choices=["step3_fast_dagger"])
    parser.add_argument("--example-limit", type=int, default=20)
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
    parser.add_argument("--output", default="priest_info_flow.json")
    parser.add_argument("--markdown", default="priest_info_flow.md")
    parser.add_argument("--run-log", default="step5_execution_heads/cards/priest/logs/2026-04-26_priest_info_flow.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut audit flux Pretre",
        expected="Mesurer quelles cartes sont vues et comment l'info est exploitee.",
        actual=f"policy={args.policy}, games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    context = prepare_policy_context("step3_fast_dagger", args)
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"Audit {config_name}",
            expected="Tracer spots Pretre et exploitations.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        configs[config_name] = evaluate_config(config_name, args.games, seed_start, args, context)
        logger.write(
            f"Audit termine {config_name}",
            expected="Reporter spots/exploit.",
            actual=(
                f"spots={configs[config_name]['summary']['spots']}, "
                f"any_exploit={configs[config_name]['summary']['overall']['any_exploit_rate']:.4f}"
            ),
            details=configs[config_name]["summary"]["overall"],
        )
    payload = {
        "created_at": now_stamp(),
        "policy": args.policy,
        "args": vars(args),
        "configs": configs,
        "aggregate": aggregate_configs(configs),
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin audit flux Pretre",
        expected="Produire JSON + markdown.",
        actual=f"json={output}, markdown={markdown}",
        details=payload["aggregate"]["overall"],
    )
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
