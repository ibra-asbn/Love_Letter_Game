"""Detailed benchmark for King usage by the evaluated policy."""

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
from step5_execution_heads.chancellor_head import load_chancellor_head
from step5_execution_heads.evaluate_combined_three_heads import Step5ThreeSeat
from step5_execution_heads.target_head import action_card


REPORT_DIR = PROJECT_ROOT / "step5_execution_heads/cards/king/reports"
LOG_DIR = PROJECT_ROOT / "step5_execution_heads/cards/king/logs"

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


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def phase_name(deck_remaining: int) -> str:
    if deck_remaining >= 11:
        return "early"
    if deck_remaining >= 6:
        return "mid"
    return "late"


def companion_for_king(hand: list[int]) -> int | None:
    if 7 not in hand:
        return None
    others = [int(card) for card in hand if int(card) != 7]
    return int(others[0]) if others else 7


def role_action(env, agent: str, obs_dict, roles: dict[str, str], policies: dict[str, object], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "model":
        return int(policies[agent].act(env, obs_dict, agent))
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


def exchange_label(my_card: int | None, target_card: int | None) -> str:
    if my_card is None or target_card is None:
        return "unknown"
    if target_card > my_card:
        return "received_higher"
    if target_card < my_card:
        return "received_lower"
    return "received_equal"


def update_king_counters(counter: Counter, event: dict, reward_eval: float, won: int) -> None:
    companion = int(event["companion"])
    phase = event["phase"]
    counter["king_hand_events"] += 1
    counter[f"hand_{companion}"] += 1
    counter[f"hand_{companion}_wins"] += won
    counter[f"phase_{phase}"] += 1
    counter[f"phase_{phase}_wins"] += won
    if event["played"]:
        label = event["exchange"]
        counter["king_played_events"] += 1
        counter[f"played_{companion}"] += 1
        counter[f"played_{companion}_wins"] += won
        counter[f"played_{phase}"] += 1
        counter[f"played_{phase}_wins"] += won
        counter[label] += 1
        counter[f"{label}_{phase}"] += 1
        counter[f"{label}_{companion}"] += 1
        target_card = event.get("target_card")
        if target_card is not None:
            counter[f"received_{int(target_card)}"] += 1
        if event.get("known_target_card") is not None:
            counter["target_known_before"] += 1
        if companion == 9:
            counter["gave_princess"] += 1
        if target_card == 9:
            counter["received_princess"] += 1


def summarize_records(records: list[dict], aggregate: Counter) -> dict:
    games = len(records)
    wins = sum(row["won"] for row in records)
    outcomes = Counter(row["outcome"] for row in records)
    king_hand_records = [row for row in records if row["king_hand"]]
    king_played_records = [row for row in records if row["king_played"]]
    played = aggregate["king_played_events"]

    phase_rows = {}
    for phase in ["early", "mid", "late"]:
        hand = aggregate[f"phase_{phase}"]
        pl = aggregate[f"played_{phase}"]
        phase_rows[phase] = {
            "hand_events": int(hand),
            "hand_winrate": float(aggregate[f"phase_{phase}_wins"] / max(1, hand)),
            "played_events": int(pl),
            "play_rate": float(pl / max(1, hand)),
            "played_winrate": float(aggregate[f"played_{phase}_wins"] / max(1, pl)),
            "received_higher_rate": float(aggregate[f"received_higher_{phase}"] / max(1, pl)),
            "received_lower_rate": float(aggregate[f"received_lower_{phase}"] / max(1, pl)),
            "received_equal_rate": float(aggregate[f"received_equal_{phase}"] / max(1, pl)),
        }

    companion_rows = {}
    for companion in range(10):
        hand = aggregate[f"hand_{companion}"]
        pl = aggregate[f"played_{companion}"]
        if hand == 0 and pl == 0:
            continue
        companion_rows[str(companion)] = {
            "name": CARD_NAMES[companion],
            "hand_events": int(hand),
            "hand_winrate": float(aggregate[f"hand_{companion}_wins"] / max(1, hand)),
            "played_events": int(pl),
            "play_rate": float(pl / max(1, hand)),
            "played_winrate": float(aggregate[f"played_{companion}_wins"] / max(1, pl)),
            "received_higher_rate": float(aggregate[f"received_higher_{companion}"] / max(1, pl)),
            "received_lower_rate": float(aggregate[f"received_lower_{companion}"] / max(1, pl)),
            "received_equal_rate": float(aggregate[f"received_equal_{companion}"] / max(1, pl)),
        }

    return {
        "games": int(games),
        "wins": int(wins),
        "winrate": float(wins / max(1, games)),
        "outcomes": {
            key: {"count": int(outcomes[key]), "rate": float(outcomes[key] / max(1, games))}
            for key in ["winner", "first_out", "second_out", "third_out", "final_loser", "unknown_loss"]
        },
        "king_hand": {
            "games": len(king_hand_records),
            "winrate": float(sum(row["won"] for row in king_hand_records) / max(1, len(king_hand_records))),
        },
        "king_played": {
            "games": len(king_played_records),
            "winrate": float(sum(row["won"] for row in king_played_records) / max(1, len(king_played_records))),
        },
        "king": {
            "hand_events": int(aggregate["king_hand_events"]),
            "played_events": int(played),
            "play_rate": float(played / max(1, aggregate["king_hand_events"])),
            "target_known_before_rate": float(aggregate["target_known_before"] / max(1, played)),
            "received_higher_rate": float(aggregate["received_higher"] / max(1, played)),
            "received_lower_rate": float(aggregate["received_lower"] / max(1, played)),
            "received_equal_rate": float(aggregate["received_equal"] / max(1, played)),
            "received_princess_rate": float(aggregate["received_princess"] / max(1, played)),
            "gave_princess_rate": float(aggregate["gave_princess"] / max(1, played)),
            "received_cards": {str(card): int(aggregate[f"received_{card}"]) for card in range(10)},
            "phase_rows": phase_rows,
            "companion_rows": companion_rows,
        },
    }


def evaluate_config(config_name: str, games: int, seed_start: int, args, context: dict) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate = Counter()

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        if args.policy == "step5_cbp":
            base_policy = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
            policy = Step5ThreeSeat(
                base_policy,
                context["chancellor_head"],
                args,
                use_chancellor=True,
                use_baron=True,
                use_prince=True,
            )
        else:
            policy = make_policy(args.policy, args, roles, eval_agent, context)
        policies = {eval_agent: policy}
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        king_events = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            action = role_action(env, agent, obs_dict, roles, policies, bot)
            pre_hand = [int(card) for card in env._hands.get(agent, [])]
            event = decode_planned_event(env, agent, action)

            if agent == eval_agent and not env._chancellor_pending:
                companion = companion_for_king(pre_hand)
                if companion is not None:
                    played_king = bool(event["kind"] == "card" and int(event["card"]) == 7)
                    target_card = event.get("target_card") if played_king else None
                    king_events.append(
                        {
                            "companion": companion,
                            "phase": phase_name(len(env._deck)),
                            "played": played_king,
                            "target_card": target_card,
                            "known_target_card": event.get("known_target_card") if played_king else None,
                            "exchange": exchange_label(companion, target_card) if played_king else "not_played",
                        }
                    )

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        for event in king_events:
            update_king_counters(aggregate, event, reward_eval, won)
        records.append(
            {
                "seed": int(seed),
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "king_hand": bool(king_events),
                "king_played": any(event["played"] for event in king_events),
            }
        )

    return summarize_records(records, aggregate)


def evaluate_policy(args, logger: ExperimentLogger, context: dict) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"Roi audit - {config_name}",
            expected="Mesurer Roi en main/joue par phase.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_config(config_name, args.games, seed_start, args, context)
        configs[config_name] = result
        logger.write(
            f"Roi audit termine {config_name}",
            expected="Reporter chaque composition.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"king_hand={result['king_hand']['winrate']:.4f}, "
                f"king_played={result['king_played']['winrate']:.4f}, "
                f"play_rate={result['king']['play_rate']:.4f}"
            ),
            details=result["king"],
        )
    return {"configs": configs, "composite": composite_score(configs)}


def weighted_conditional(policy: dict, key: str) -> dict:
    games = 0
    wins = 0.0
    for config in policy["configs"].values():
        row = config[key]
        n = int(row["games"])
        games += n
        wins += float(row["winrate"]) * n
    return {"games": games, "winrate": float(wins / max(1, games))}


def aggregate_king(policy: dict) -> dict:
    total = Counter()
    phase = defaultdict(Counter)
    companion = defaultdict(Counter)
    for config in policy["configs"].values():
        king = config["king"]
        hand = int(king["hand_events"])
        played = int(king["played_events"])
        total["hand"] += hand
        total["played"] += played
        for key in [
            "target_known_before_rate",
            "received_higher_rate",
            "received_lower_rate",
            "received_equal_rate",
            "received_princess_rate",
            "gave_princess_rate",
        ]:
            total[key] += king[key] * played
        for name, row in king["phase_rows"].items():
            h = int(row["hand_events"])
            pl = int(row["played_events"])
            phase[name]["hand"] += h
            phase[name]["hand_wins"] += row["hand_winrate"] * h
            phase[name]["played"] += pl
            phase[name]["played_wins"] += row["played_winrate"] * pl
            phase[name]["received_higher"] += row["received_higher_rate"] * pl
            phase[name]["received_lower"] += row["received_lower_rate"] * pl
            phase[name]["received_equal"] += row["received_equal_rate"] * pl
        for card, row in king["companion_rows"].items():
            h = int(row["hand_events"])
            pl = int(row["played_events"])
            companion[card]["hand"] += h
            companion[card]["hand_wins"] += row["hand_winrate"] * h
            companion[card]["played"] += pl
            companion[card]["played_wins"] += row["played_winrate"] * pl
            companion[card]["received_higher"] += row["received_higher_rate"] * pl
            companion[card]["received_lower"] += row["received_lower_rate"] * pl
            companion[card]["received_equal"] += row["received_equal_rate"] * pl

    phase_rows = {}
    for name in ["early", "mid", "late"]:
        row = phase[name]
        h = row["hand"]
        pl = row["played"]
        phase_rows[name] = {
            "hand_events": int(h),
            "hand_winrate": float(row["hand_wins"] / max(1, h)),
            "played_events": int(pl),
            "play_rate": float(pl / max(1, h)),
            "played_winrate": float(row["played_wins"] / max(1, pl)),
            "received_higher_rate": float(row["received_higher"] / max(1, pl)),
            "received_lower_rate": float(row["received_lower"] / max(1, pl)),
            "received_equal_rate": float(row["received_equal"] / max(1, pl)),
        }

    companion_rows = {}
    for card, row in companion.items():
        h = row["hand"]
        pl = row["played"]
        companion_rows[card] = {
            "name": CARD_NAMES[int(card)],
            "hand_events": int(h),
            "hand_winrate": float(row["hand_wins"] / max(1, h)),
            "played_events": int(pl),
            "play_rate": float(pl / max(1, h)),
            "played_winrate": float(row["played_wins"] / max(1, pl)),
            "received_higher_rate": float(row["received_higher"] / max(1, pl)),
            "received_lower_rate": float(row["received_lower"] / max(1, pl)),
            "received_equal_rate": float(row["received_equal"] / max(1, pl)),
        }

    played_total = total["played"]
    return {
        "hand_events": int(total["hand"]),
        "played_events": int(played_total),
        "play_rate": float(played_total / max(1, total["hand"])),
        "target_known_before_rate": float(total["target_known_before_rate"] / max(1, played_total)),
        "received_higher_rate": float(total["received_higher_rate"] / max(1, played_total)),
        "received_lower_rate": float(total["received_lower_rate"] / max(1, played_total)),
        "received_equal_rate": float(total["received_equal_rate"] / max(1, played_total)),
        "received_princess_rate": float(total["received_princess_rate"] / max(1, played_total)),
        "gave_princess_rate": float(total["gave_princess_rate"] / max(1, played_total)),
        "phase_rows": phase_rows,
        "companion_rows": companion_rows,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    policy = payload["policy"]
    king = aggregate_king(policy)
    hand = weighted_conditional(policy, "king_hand")
    played = weighted_conditional(policy, "king_played")
    lines = [
        "# Roi - Benchmark Detaille",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Policy: `{payload['args']['policy']}`.",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "## Synthese",
        "",
        "| Metrique | Valeur |",
        "|---|---:|",
        f"| Composite arena | {policy['composite']:.5f} |",
        f"| Roi en main | {pct(hand['winrate'])} (n={hand['games']}) |",
        f"| Roi joue | {pct(played['winrate'])} (n={played['games']}) |",
        f"| Utilisation du Roi par occurrence | {pct(king['play_rate'])} |",
        f"| Cible connue avant echange | {pct(king['target_known_before_rate'])} |",
        f"| Recoit carte plus haute | {pct(king['received_higher_rate'])} |",
        f"| Recoit carte plus basse | {pct(king['received_lower_rate'])} |",
        f"| Recoit carte egale | {pct(king['received_equal_rate'])} |",
        f"| Recoit Princesse | {pct(king['received_princess_rate'])} |",
        f"| Donne Princesse | {pct(king['gave_princess_rate'])} |",
        "",
        "## Par Composition D'Arene",
        "",
        "| Composition | Winrate global | Roi en main | Roi joue | Utilisation Roi |",
        "|---|---:|---:|---:|---:|",
    ]
    for config_name, config in policy["configs"].items():
        lines.append(
            f"| {CONFIG_LABELS[config_name]} | {pct(config['winrate'])} | "
            f"{pct(config['king_hand']['winrate'])} (n={config['king_hand']['games']}) | "
            f"{pct(config['king_played']['winrate'])} (n={config['king_played']['games']}) | "
            f"{pct(config['king']['play_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Par Moment De Partie",
            "",
            "| Phase | Roi en main events | Winrate avec Roi | Roi joue events | Pct utilisation | Winrate si joue | Recoit + haut | Recoit + bas |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for phase in ["early", "mid", "late"]:
        row = king["phase_rows"][phase]
        lines.append(
            f"| {phase} | {row['hand_events']} | {pct(row['hand_winrate'])} | "
            f"{row['played_events']} | {pct(row['play_rate'])} | {pct(row['played_winrate'])} | "
            f"{pct(row['received_higher_rate'])} | {pct(row['received_lower_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Par Carte Accompagnante",
            "",
            "| Carte avec Roi | Occurrences | Winrate avec Roi | Pct joue Roi | Winrate si joue | Recoit + haut | Recoit + bas |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for card, row in sorted(king["companion_rows"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {card} {row['name']} | {row['hand_events']} | {pct(row['hand_winrate'])} | "
            f"{pct(row['play_rate'])} | {pct(row['played_winrate'])} | "
            f"{pct(row['received_higher_rate'])} | {pct(row['received_lower_rate'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Detailed King benchmark.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=3200000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--policy", default="step3_fast_dagger", choices=["step3_fast_dagger", "step5_cbp"])
    parser.add_argument("--chancellor-head", default="step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth")
    parser.add_argument("--chancellor-margin", type=float, default=0.10)
    parser.add_argument("--retarget-margin", type=float, default=0.10)
    parser.add_argument("--veto-score", type=float, default=0.05)
    parser.add_argument("--force-score", type=float, default=0.32)
    parser.add_argument("--self-force-score", type=float, default=0.55)
    parser.add_argument("--min-princess-prob", type=float, default=0.24)
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
    parser.add_argument("--output", default="king_usage_benchmark.json")
    parser.add_argument("--markdown", default="king_usage_benchmark.md")
    parser.add_argument("--run-log", default="step5_execution_heads/cards/king/logs/2026-04-26_king_usage_benchmark.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut benchmark Roi",
        expected="Mesurer Roi en main/joue, phases et qualite brute des echanges.",
        actual=f"policy={args.policy}, games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    context = prepare_policy_context("step3_fast_dagger", args)
    if args.policy == "step5_cbp":
        head_path = PROJECT_ROOT / args.chancellor_head
        context["chancellor_head"], _ckpt = load_chancellor_head(head_path, args.device)
    policy = evaluate_policy(args, logger, context)
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "policy": policy,
        "summary": {
            "king_hand": weighted_conditional(policy, "king_hand"),
            "king_played": weighted_conditional(policy, "king_played"),
            "king": aggregate_king(policy),
        },
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin benchmark Roi",
        expected="Produire JSON + rapport markdown.",
        actual=f"json={output}, markdown={markdown}",
        details=payload["summary"],
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
