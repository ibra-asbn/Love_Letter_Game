"""Evaluate a local Baron action-value specialist.

The specialist is deliberately scoped: it only wakes up when Baron is in hand.
It can either keep Step3's decision, retarget a Baron action, or choose the
other card when the Baron duel is too risky.
"""

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
    classify_outcome,
    decode_planned_event,
    make_policy,
    prepare_policy_context,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step2_rl_finetune.evaluate_step2 import random_action
from step5_execution_heads.target_head import action_card, action_target, infer_kept_card, target_distribution_from_obs


REPORT_DIR = PROJECT_ROOT / "step5_execution_heads/cards/baron/reports"
LOG_DIR = PROJECT_ROOT / "step5_execution_heads/cards/baron/logs"

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


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def same_card_actions(mask: np.ndarray, card: int) -> list[int]:
    return [action for action in valid_actions(mask) if action_card(action) == int(card)]


def companion_for_baron(hand: list[int]) -> int | None:
    if 3 not in hand:
        return None
    others = [int(card) for card in hand if int(card) != 3]
    return int(others[0]) if others else 3


def baron_duel_stats(env, obs_dict: dict, agent: str, action: int) -> dict:
    kept = infer_kept_card(3, list(env._hands.get(agent, [])))
    target = action_target(action)
    dist = target_distribution_from_obs(obs_dict["observation"], agent, target)
    probs = dist["probs"]
    p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
    p_equal = float(probs[kept]) if 0 <= kept < 10 else 0.0
    p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
    expected = float(np.dot(probs, np.arange(10, dtype=np.float32)))
    reliable = bool(dist["known_flag"] or dist["public_min"] > 0.0)
    score = p_lower - 1.55 * p_higher - 0.10 * p_equal
    if dist["known_flag"]:
        score += 0.30 * (p_lower - 1.50 * p_higher - 0.10 * p_equal)
    if kept >= 8:
        score += 0.04 * (expected / 9.0)
    elif kept <= 5:
        score -= 0.15 * p_higher
    return {
        "kept": kept,
        "target": target,
        "score": float(score),
        "p_lower": p_lower,
        "p_equal": p_equal,
        "p_higher": p_higher,
        "expected": expected,
        "known": bool(dist["known_flag"]),
        "reliable": reliable,
    }


def best_baron_action(env, obs_dict: dict, agent: str) -> tuple[int | None, dict | None]:
    candidates = same_card_actions(obs_dict["action_mask"], 3)
    if not candidates:
        return None, None
    scored = [(action, baron_duel_stats(env, obs_dict, agent, action)) for action in candidates]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    return int(scored[0][0]), scored[0][1]


def guard_action(env, agent: str, mask: np.ndarray, candidates: list[int]) -> int:
    max_counts = np.asarray([2, 6, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.float32)
    played = np.zeros(10, dtype=np.float32)
    for seat in env.possible_agents:
        for card in env._played_cards.get(seat, []):
            played[card] += 1.0
    remaining = max_counts - played
    for card in env._hands.get(agent, []):
        remaining[card] -= 1.0
    remaining = np.clip(remaining, 0.0, None)
    remaining[1] = 0.0
    guess_order = list(np.argsort(remaining)[::-1])
    target_order = [action_target(action) for action in candidates]
    for target in target_order:
        for guess in guess_order:
            action = 100 + int(target) * 10 + int(guess)
            if mask[action] == 1:
                return int(action)
    return int(candidates[0])


def priest_action(env, agent: str, obs_dict: dict, candidates: list[int]) -> int:
    rows = []
    for action in candidates:
        target = action_target(action)
        dist = target_distribution_from_obs(obs_dict["observation"], agent, target)
        probs = dist["probs"]
        entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
        known_penalty = 1.0 if dist["known_flag"] else 0.0
        rows.append((entropy - known_penalty, action))
    rows.sort(reverse=True)
    return int(rows[0][1])


def prince_action(env, agent: str, mask: np.ndarray, candidates: list[int]) -> int:
    my_idx = env.possible_agents.index(agent)
    for action in candidates:
        target = action_target(action)
        target_name = env.possible_agents[target] if target < len(env.possible_agents) else None
        if target_name and env._known_cards[agent][target_name][9] >= 1.0:
            return int(action)
    self_action = 500 + my_idx * 10
    if mask[self_action] == 1:
        return int(self_action)
    return int(candidates[0])


def alternative_action(env, obs_dict: dict, agent: str, companion: int) -> int | None:
    mask = obs_dict["action_mask"]
    candidates = same_card_actions(mask, companion)
    if not candidates:
        return None
    if companion == 1:
        return guard_action(env, agent, mask, candidates)
    if companion == 2:
        return priest_action(env, agent, obs_dict, candidates)
    if companion == 5:
        return prince_action(env, agent, mask, candidates)
    return int(candidates[0])


def should_play_baron(companion: int, stats: dict, base_played_baron: bool) -> bool:
    if companion == 3:
        return True
    if companion >= 8:
        return True
    if companion == 7:
        return stats["p_higher"] <= 0.24 or stats["score"] >= 0.18 or stats["reliable"]
    if companion == 6:
        return stats["p_higher"] <= 0.24 or stats["score"] >= 0.24 or (stats["reliable"] and stats["p_higher"] <= 0.34)
    if companion == 5:
        return stats["p_higher"] <= 0.20 or stats["score"] >= 0.30 or (stats["reliable"] and stats["p_higher"] <= 0.30)
    return stats["reliable"] and stats["p_higher"] <= 0.08 and stats["p_lower"] >= 0.70


class BaronSpecialistSeat:
    def __init__(self, base_policy, mode: str, args):
        self.base_policy = base_policy
        self.mode = mode
        self.stats = Counter()
        self.examples = []
        self.example_limit = args.example_limit

    def act(self, env, obs_dict, agent: str) -> int:
        base_action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return base_action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_baron(hand)
        if companion is None:
            return base_action

        self.stats["baron_hand_checks"] += 1
        base_played_baron = action_card(base_action) == 3
        if base_played_baron:
            self.stats["base_baron_plays"] += 1

        if self.mode == "baseline":
            return base_action

        best_action, best_stats = best_baron_action(env, obs_dict, agent)
        if best_action is None or best_stats is None:
            return base_action

        if self.mode == "random_target":
            if not base_played_baron:
                return base_action
            candidates = same_card_actions(obs_dict["action_mask"], 3)
            new_action = int(np.random.choice(np.asarray(candidates, dtype=np.int64)))
            if new_action != base_action:
                self.stats["overrides"] += 1
            return new_action

        alt = alternative_action(env, obs_dict, agent, companion)
        play_baron = should_play_baron(companion, best_stats, base_played_baron)
        chosen = int(best_action) if play_baron else (int(alt) if alt is not None else base_action)

        # Avoid forcing new Baron plays from very low-information non-Baron base
        # decisions. The first goal is to repair risky Baron choices, not to make
        # the player more reckless.
        if not base_played_baron and action_card(chosen) == 3 and not best_stats["reliable"]:
            chosen = base_action

        if chosen != base_action:
            self.stats["overrides"] += 1
            if base_played_baron and action_card(chosen) != 3:
                self.stats["baron_to_other"] += 1
            elif not base_played_baron and action_card(chosen) == 3:
                self.stats["other_to_baron"] += 1
            else:
                self.stats["baron_retarget"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "hand": hand,
                        "companion": companion,
                        "base_action": base_action,
                        "chosen": chosen,
                        "best_baron": best_action,
                        "alt": alt,
                        "stats": best_stats,
                    }
                )
        return int(chosen)


def make_baron_policy(args, roles, eval_agent, context, mode: str):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return BaronSpecialistSeat(base, mode, args)


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


def update_baron_counters(counter: Counter, event: dict, reward_eval: float, won: int) -> None:
    companion = event["companion"]
    counter["baron_hand_games"] += 1
    counter[f"hand_{companion}"] += 1
    counter[f"hand_{companion}_wins"] += won
    if event["played"]:
        counter["baron_played_games"] += 1
        counter[f"played_{companion}"] += 1
        counter[f"played_{companion}_wins"] += won
        counter[f"played_{companion}_reward_sum"] += reward_eval
        counter[event["duel"]] += 1
        counter[f"{event['duel']}_{companion}"] += 1


def summarize_records(records: list[dict], aggregate: Counter, policy_stats: Counter, examples: list[dict]) -> dict:
    games = len(records)
    wins = sum(row["won"] for row in records)
    rewards = np.asarray([row["reward"] for row in records], dtype=np.float32)
    outcomes = Counter(row["outcome"] for row in records)
    baron_hand_records = [row for row in records if row["baron_hand"]]
    baron_played_records = [row for row in records if row["baron_played"]]
    companion_rows = {}
    for companion in range(10):
        hand = aggregate[f"hand_{companion}"]
        played = aggregate[f"played_{companion}"]
        if hand == 0 and played == 0:
            continue
        companion_rows[str(companion)] = {
            "name": CARD_NAMES[companion],
            "hand_games": int(hand),
            "hand_winrate": float(aggregate[f"hand_{companion}_wins"] / max(1, hand)),
            "played": int(played),
            "play_rate": float(played / max(1, hand)),
            "played_winrate": float(aggregate[f"played_{companion}_wins"] / max(1, played)),
            "played_mean_reward": float(aggregate[f"played_{companion}_reward_sum"] / max(1, played)),
            "duel_win_rate": float(aggregate[f"duel_win_{companion}"] / max(1, played)),
            "duel_loss_rate": float(aggregate[f"duel_loss_{companion}"] / max(1, played)),
            "duel_tie_rate": float(aggregate[f"duel_tie_{companion}"] / max(1, played)),
            "duel_no_target_rate": float(aggregate[f"duel_no_target_{companion}"] / max(1, played)),
        }
    return {
        "games": games,
        "wins": int(wins),
        "winrate": float(wins / max(1, games)),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "outcomes": {key: {"count": int(outcomes[key]), "rate": float(outcomes[key] / max(1, games))} for key in sorted(outcomes)},
        "baron_hand": {
            "games": len(baron_hand_records),
            "winrate": float(sum(row["won"] for row in baron_hand_records) / max(1, len(baron_hand_records))),
        },
        "baron_played": {
            "games": len(baron_played_records),
            "winrate": float(sum(row["won"] for row in baron_played_records) / max(1, len(baron_played_records))),
        },
        "baron": {
            "duel_win_rate": float(aggregate["duel_win"] / max(1, aggregate["baron_played_games"])),
            "duel_loss_rate": float(aggregate["duel_loss"] / max(1, aggregate["baron_played_games"])),
            "duel_tie_rate": float(aggregate["duel_tie"] / max(1, aggregate["baron_played_games"])),
            "companion_rows": companion_rows,
        },
        "specialist": {
            "raw_counts": dict(policy_stats),
            "override_rate": float(policy_stats["overrides"] / max(1, policy_stats["baron_hand_checks"])),
            "examples": examples,
        },
    }


def evaluate_policy_config(policy_name: str, mode: str, config_name: str, games: int, seed_start: int, args, context: dict) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate = Counter()
    policy_stats = Counter()
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policy = make_baron_policy(args, roles, eval_agent, context, mode)
        policies = {eval_agent: policy}
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        baron_events = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            if roles[agent] == "model":
                action = policy.act(env, obs_dict, agent)
            elif roles[agent] == "heuristic":
                action = int(bot.choose_action(env, agent))
            elif roles[agent] == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(roles[agent])

            pre_hand = [int(card) for card in env._hands.get(agent, [])]
            event = decode_planned_event(env, agent, action)
            if agent == eval_agent and not env._chancellor_pending:
                companion = companion_for_baron(pre_hand)
                if companion is not None:
                    played_baron = bool(event["kind"] == "card" and int(event["card"]) == 3)
                    duel = "not_played"
                    if played_baron:
                        target_card = event.get("target_card")
                        remaining = event.get("remaining_hand", [])
                        kept = int(remaining[0]) if remaining else companion
                        if target_card is None:
                            duel = "duel_no_target"
                        elif kept > int(target_card):
                            duel = "duel_win"
                        elif int(target_card) > kept:
                            duel = "duel_loss"
                        else:
                            duel = "duel_tie"
                    baron_events.append({"companion": companion, "played": played_baron, "duel": duel})

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        for event in baron_events:
            update_baron_counters(aggregate, event, reward_eval, won)
        if isinstance(policy, BaronSpecialistSeat):
            policy_stats.update(policy.stats)
            examples.extend(policy.examples[: max(0, args.example_limit - len(examples))])
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "baron_hand": bool(baron_events),
                "baron_played": any(event["played"] for event in baron_events),
            }
        )
    return summarize_records(records, aggregate, policy_stats, examples[: args.example_limit])


def evaluate_policy(policy_name: str, mode: str, args, logger: ExperimentLogger, context: dict) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Evaluer Step5 Baron action-value par composition.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, mode, config_name, args.games, seed_start, args, context)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque etape terminee.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"baron_hand={result['baron_hand']['winrate']:.4f}, "
                f"duel_loss={result['baron']['duel_loss_rate']:.4f}"
            ),
            details=result,
        )
    return {"configs": configs, "composite": composite_score(configs)}


def weighted_conditional(policy: dict, key: str) -> dict:
    games = 0
    wins = 0.0
    for config in policy["configs"].values():
        row = config[key]
        games += int(row["games"])
        wins += float(row["winrate"]) * int(row["games"])
    return {"games": games, "winrate": float(wins / max(1, games))}


def aggregate_baron(policy: dict) -> dict:
    total = Counter()
    companion = defaultdict(Counter)
    for config in policy["configs"].values():
        baron = config["baron"]
        played = config["baron_played"]["games"]
        total["played"] += played
        total["duel_win"] += baron["duel_win_rate"] * played
        total["duel_loss"] += baron["duel_loss_rate"] * played
        total["duel_tie"] += baron["duel_tie_rate"] * played
        for card, row in baron["companion_rows"].items():
            hand = row["hand_games"]
            pl = row["played"]
            companion[card]["hand"] += hand
            companion[card]["hand_wins"] += row["hand_winrate"] * hand
            companion[card]["played"] += pl
            companion[card]["played_wins"] += row["played_winrate"] * pl
            companion[card]["duel_win"] += row["duel_win_rate"] * pl
            companion[card]["duel_loss"] += row["duel_loss_rate"] * pl
    rows = {}
    for card, row in companion.items():
        rows[card] = {
            "name": CARD_NAMES[int(card)],
            "hand_games": int(row["hand"]),
            "hand_winrate": float(row["hand_wins"] / max(1, row["hand"])),
            "played": int(row["played"]),
            "play_rate": float(row["played"] / max(1, row["hand"])),
            "played_winrate": float(row["played_wins"] / max(1, row["played"])),
            "duel_win_rate": float(row["duel_win"] / max(1, row["played"])),
            "duel_loss_rate": float(row["duel_loss"] / max(1, row["played"])),
        }
    return {
        "duel_win_rate": float(total["duel_win"] / max(1, total["played"])),
        "duel_loss_rate": float(total["duel_loss"] / max(1, total["played"])),
        "duel_tie_rate": float(total["duel_tie"] / max(1, total["played"])),
        "companion_rows": rows,
    }


def write_markdown(payload: dict, path: Path) -> None:
    policies = payload["policies"]
    labels = {
        "baseline": "Step3 rapide",
        "random_target": "Baron target random",
        "specialist": "Step5 Baron specialist",
    }
    lines = [
        "# Step5 Baron - Evaluation Action-Value",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "## Synthese",
        "",
        "| Politique | Composite | Baron en main | Baron joue | Duel gagne | Duel perdu |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "random_target", "specialist"]:
        policy = policies[name]
        hand = weighted_conditional(policy, "baron_hand")
        played = weighted_conditional(policy, "baron_played")
        baron = aggregate_baron(policy)
        lines.append(
            f"| {labels[name]} | {policy['composite']:.5f} | "
            f"{100.0 * hand['winrate']:.2f}% (n={hand['games']}) | "
            f"{100.0 * played['winrate']:.2f}% (n={played['games']}) | "
            f"{100.0 * baron['duel_win_rate']:.2f}% | "
            f"{100.0 * baron['duel_loss_rate']:.2f}% |"
        )
    lines.extend(["", "## Par Carte Accompagnante", ""])
    for name in ["baseline", "specialist"]:
        lines.extend(
            [
                f"### {labels[name]}",
                "",
                "| Carte avec Baron | Baron en main | Pct joue Baron | Winrate si main | Winrate si joue | Duel gagne | Duel perdu |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for card, row in sorted(aggregate_baron(policies[name])["companion_rows"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {card} {row['name']} | {row['hand_games']} | "
                f"{100.0 * row['play_rate']:.2f}% | "
                f"{100.0 * row['hand_winrate']:.2f}% | "
                f"{100.0 * row['played_winrate']:.2f}% | "
                f"{100.0 * row['duel_win_rate']:.2f}% | "
                f"{100.0 * row['duel_loss_rate']:.2f}% |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step5 Baron action-value specialist.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1900000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--example-limit", type=int, default=24)
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
    parser.add_argument("--output", default="baron_specialist_eval.json")
    parser.add_argument("--markdown", default="baron_specialist_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/cards/baron/logs/2026-04-26_baron_specialist_eval.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step5 Baron action-value",
        expected="Comparer Step3, Baron target random et specialiste local.",
        actual=f"games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    context = prepare_policy_context("step3_fast_dagger", args)
    policies = {
        "baseline": evaluate_policy("baseline", "baseline", args, logger, context),
        "random_target": evaluate_policy("random_target", "random_target", args, logger, context),
        "specialist": evaluate_policy("specialist", "specialist", args, logger, context),
    }
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "policies": policies,
        "summary": {
            name: {
                "composite": policy["composite"],
                "baron_hand": weighted_conditional(policy, "baron_hand"),
                "baron_played": weighted_conditional(policy, "baron_played"),
                "baron": aggregate_baron(policy),
            }
            for name, policy in policies.items()
        },
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step5 Baron action-value",
        expected="Obtenir une decision sur la validite du specialiste.",
        actual=f"json={output}, markdown={markdown}",
        details=payload["summary"],
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
