"""Evaluate a local Prince action-value specialist.

The specialist only wakes up when the evaluated Step3 player has Prince in hand.
The first goal is diagnostic: compare Step3 alone with Step3+Prince on games
where Prince actually appeared in the evaluated player's hand.
"""

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
from step5_execution_heads.target_head import action_card, action_target, infer_kept_card, target_distribution_from_obs


REPORT_DIR = PROJECT_ROOT / "step5_execution_heads/cards/prince/reports"
LOG_DIR = PROJECT_ROOT / "step5_execution_heads/cards/prince/logs"

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


def companion_for_prince(hand: list[int]) -> int | None:
    if 5 not in hand:
        return None
    others = [int(card) for card in hand if int(card) != 5]
    return int(others[0]) if others else 5


def phase_name(deck_remaining: int) -> str:
    if deck_remaining >= 11:
        return "early"
    if deck_remaining >= 6:
        return "mid"
    return "late"


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


def baron_action(env, agent: str, obs_dict: dict, candidates: list[int]) -> int:
    rows = []
    for action in candidates:
        kept = infer_kept_card(3, list(env._hands.get(agent, [])))
        target = action_target(action)
        probs = target_distribution_from_obs(obs_dict["observation"], agent, target)["probs"]
        p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
        p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
        rows.append((p_lower - 1.5 * p_higher, action))
    rows.sort(reverse=True)
    return int(rows[0][1])


def alternative_action(env, obs_dict: dict, agent: str, companion: int) -> int | None:
    if companion == 9:
        return None
    mask = obs_dict["action_mask"]
    candidates = same_card_actions(mask, companion)
    if not candidates:
        return None
    if companion == 1:
        return guard_action(env, agent, mask, candidates)
    if companion == 2:
        return priest_action(env, agent, obs_dict, candidates)
    if companion == 3:
        return baron_action(env, agent, obs_dict, candidates)
    return int(candidates[0])


def prince_action_stats(env, obs_dict: dict, agent: str, action: int, companion: int) -> dict:
    target = action_target(action)
    my_idx = env.possible_agents.index(agent)
    deck_remaining = len(env._deck)
    if target == my_idx:
        if companion == 9:
            score = -5.0
        elif companion <= 1:
            score = 0.82
        elif companion == 2:
            score = 0.62
        elif companion == 3:
            score = 0.28
        elif companion == 4:
            score = 0.12
        elif companion == 5:
            score = -0.02
        elif companion == 6:
            score = -0.08
        elif companion == 7:
            score = -0.24
        else:
            score = -0.36
        if deck_remaining <= 5 and companion <= 2:
            score -= 0.08
        return {
            "target": target,
            "self_target": True,
            "score": float(score),
            "p_princess": 1.0 if companion == 9 else 0.0,
            "p_countess": 1.0 if companion == 8 else 0.0,
            "p_high": 1.0 if companion >= 7 else 0.0,
            "expected": float(companion),
            "known": True,
            "reliable": True,
        }

    dist = target_distribution_from_obs(obs_dict["observation"], agent, target)
    probs = dist["probs"]
    p_princess = float(probs[9])
    p_countess = float(probs[8])
    p_king = float(probs[7])
    p_high = float(probs[7:].sum())
    p_low = float(probs[:4].sum())
    expected = float(np.dot(probs, np.arange(10, dtype=np.float32)))
    score = 2.90 * p_princess + 0.58 * p_countess + 0.24 * p_king + 0.06 * (expected / 9.0) - 0.10 * p_low
    if dist["known_flag"] and p_princess >= 0.99:
        score += 1.15
    elif dist["known_flag"] and expected <= 3.0:
        score -= 0.35
    if deck_remaining <= 5:
        score += 0.14 * p_high
    return {
        "target": target,
        "self_target": False,
        "score": float(score),
        "p_princess": p_princess,
        "p_countess": p_countess,
        "p_high": p_high,
        "expected": expected,
        "known": bool(dist["known_flag"]),
        "reliable": bool(dist["known_flag"] or dist["public_min"] > 0.0 or p_princess >= 0.20),
    }


def best_prince_action(env, obs_dict: dict, agent: str, companion: int) -> tuple[int | None, dict | None]:
    candidates = same_card_actions(obs_dict["action_mask"], 5)
    if not candidates:
        return None, None
    scored = [(action, prince_action_stats(env, obs_dict, agent, action, companion)) for action in candidates]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    return int(scored[0][0]), scored[0][1]


def should_force_prince(stats: dict, companion: int, args) -> bool:
    if stats["self_target"]:
        return companion <= 2 and stats["score"] >= args.self_force_score
    if stats["known"] and stats["p_princess"] >= 0.99:
        return True
    if stats["p_princess"] >= args.min_princess_prob and stats["score"] >= args.force_score:
        return True
    if companion == 9 and not stats["self_target"] and stats["score"] >= 0.10:
        return True
    return False


class PrinceSpecialistSeat:
    def __init__(self, base_policy, mode: str, args):
        self.base_policy = base_policy
        self.mode = mode
        self.stats = Counter()
        self.examples = []
        self.example_limit = args.example_limit
        self.args = args

    def act(self, env, obs_dict, agent: str) -> int:
        base_action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return base_action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_prince(hand)
        if companion is None:
            return base_action

        self.stats["prince_hand_checks"] += 1
        base_played_prince = action_card(base_action) == 5
        if base_played_prince:
            self.stats["base_prince_plays"] += 1

        if self.mode == "baseline":
            return base_action

        best_action, best_stats = best_prince_action(env, obs_dict, agent, companion)
        if best_action is None or best_stats is None:
            return base_action

        chosen = base_action
        current_stats = None
        if base_played_prince:
            current_stats = prince_action_stats(env, obs_dict, agent, base_action, companion)
            if best_action != base_action and best_stats["score"] - current_stats["score"] >= self.args.retarget_margin:
                chosen = int(best_action)
            elif best_stats["score"] < self.args.veto_score and companion in {6, 7, 8}:
                alt = alternative_action(env, obs_dict, agent, companion)
                if alt is not None:
                    chosen = int(alt)
        elif should_force_prince(best_stats, companion, self.args):
            chosen = int(best_action)

        if chosen != base_action:
            self.stats["overrides"] += 1
            if base_played_prince and action_card(chosen) == 5:
                self.stats["prince_retarget"] += 1
            elif base_played_prince and action_card(chosen) != 5:
                self.stats["prince_to_other"] += 1
            elif not base_played_prince and action_card(chosen) == 5:
                self.stats["other_to_prince"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "hand": hand,
                        "companion": companion,
                        "base_action": base_action,
                        "chosen": chosen,
                        "best_prince": best_action,
                        "best_stats": best_stats,
                        "current_stats": current_stats,
                    }
                )
        return int(chosen)


def make_prince_policy(args, roles, eval_agent, context, mode: str):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return PrinceSpecialistSeat(base, mode, args)


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


def update_prince_counters(counter: Counter, event: dict, reward_eval: float, won: int) -> None:
    companion = event["companion"]
    phase = event["phase"]
    counter["prince_hand_events"] += 1
    counter[f"hand_{companion}"] += 1
    counter[f"hand_{companion}_wins"] += won
    counter[f"phase_{phase}"] += 1
    counter[f"phase_{phase}_wins"] += won
    if event["played"]:
        counter["prince_played_events"] += 1
        counter[f"played_{companion}"] += 1
        counter[f"played_{companion}_wins"] += won
        counter[f"played_{phase}"] += 1
        counter[f"played_{phase}_wins"] += won
        counter[f"target_{event['target_kind']}"] += 1
        if event["princess_hit"]:
            counter["princess_hits"] += 1
            counter[f"princess_hits_{companion}"] += 1
        if event["self_suicide"]:
            counter["self_suicides"] += 1


def summarize_records(records: list[dict], aggregate: Counter, policy_stats: Counter, examples: list[dict]) -> dict:
    games = len(records)
    wins = sum(row["won"] for row in records)
    rewards = np.asarray([row["reward"] for row in records], dtype=np.float32)
    outcomes = Counter(row["outcome"] for row in records)
    prince_hand_records = [row for row in records if row["prince_hand"]]
    prince_played_records = [row for row in records if row["prince_played"]]
    companion_rows = {}
    for companion in range(10):
        hand = aggregate[f"hand_{companion}"]
        played = aggregate[f"played_{companion}"]
        if hand == 0 and played == 0:
            continue
        companion_rows[str(companion)] = {
            "name": CARD_NAMES[companion],
            "hand_events": int(hand),
            "hand_winrate": float(aggregate[f"hand_{companion}_wins"] / max(1, hand)),
            "played_events": int(played),
            "play_rate": float(played / max(1, hand)),
            "played_winrate": float(aggregate[f"played_{companion}_wins"] / max(1, played)),
            "princess_hit_rate": float(aggregate[f"princess_hits_{companion}"] / max(1, played)),
        }
    phase_rows = {}
    for phase in ["early", "mid", "late"]:
        hand = aggregate[f"phase_{phase}"]
        played = aggregate[f"played_{phase}"]
        phase_rows[phase] = {
            "hand_events": int(hand),
            "hand_winrate": float(aggregate[f"phase_{phase}_wins"] / max(1, hand)),
            "played_events": int(played),
            "play_rate": float(played / max(1, hand)),
            "played_winrate": float(aggregate[f"played_{phase}_wins"] / max(1, played)),
        }
    played = aggregate["prince_played_events"]
    return {
        "games": games,
        "wins": int(wins),
        "winrate": float(wins / max(1, games)),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "outcomes": {key: {"count": int(outcomes[key]), "rate": float(outcomes[key] / max(1, games))} for key in sorted(outcomes)},
        "prince_hand": {
            "games": len(prince_hand_records),
            "winrate": float(sum(row["won"] for row in prince_hand_records) / max(1, len(prince_hand_records))),
        },
        "prince_played": {
            "games": len(prince_played_records),
            "winrate": float(sum(row["won"] for row in prince_played_records) / max(1, len(prince_played_records))),
        },
        "prince": {
            "hand_events": int(aggregate["prince_hand_events"]),
            "played_events": int(played),
            "play_rate": float(played / max(1, aggregate["prince_hand_events"])),
            "target_self_rate": float(aggregate["target_self"] / max(1, played)),
            "target_opponent_rate": float(aggregate["target_opponent"] / max(1, played)),
            "princess_hit_rate": float(aggregate["princess_hits"] / max(1, played)),
            "self_suicide_rate": float(aggregate["self_suicides"] / max(1, played)),
            "companion_rows": companion_rows,
            "phase_rows": phase_rows,
        },
        "specialist": {
            "raw_counts": dict(policy_stats),
            "override_rate": float(policy_stats["overrides"] / max(1, policy_stats["prince_hand_checks"])),
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
        policy = make_prince_policy(args, roles, eval_agent, context, mode)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        prince_events = []

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
                companion = companion_for_prince(pre_hand)
                if companion is not None:
                    played_prince = bool(event["kind"] == "card" and int(event["card"]) == 5)
                    target_kind = "none"
                    princess_hit = False
                    self_suicide = False
                    if played_prince:
                        if event.get("target") == eval_agent:
                            target_kind = "self"
                        elif event.get("target"):
                            target_kind = "opponent"
                        princess_hit = bool(event.get("target_card") == 9)
                        self_suicide = bool(event.get("target") == eval_agent and event.get("target_card") == 9)
                    prince_events.append(
                        {
                            "companion": companion,
                            "phase": phase_name(len(env._deck)),
                            "played": played_prince,
                            "target_kind": target_kind,
                            "princess_hit": princess_hit,
                            "self_suicide": self_suicide,
                        }
                    )

            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        for event in prince_events:
            update_prince_counters(aggregate, event, reward_eval, won)
        if isinstance(policy, PrinceSpecialistSeat):
            policy_stats.update(policy.stats)
            examples.extend(policy.examples[: max(0, args.example_limit - len(examples))])
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "prince_hand": bool(prince_events),
                "prince_played": any(event["played"] for event in prince_events),
            }
        )
    return summarize_records(records, aggregate, policy_stats, examples[: args.example_limit])


def evaluate_policy(policy_name: str, mode: str, args, logger: ExperimentLogger, context: dict) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Evaluer Step5 Prince par composition, en gardant le filtre Prince en main.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, mode, config_name, args.games, seed_start, args, context)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque etape terminee.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"prince_hand={result['prince_hand']['winrate']:.4f}, "
                f"princess_hit={result['prince']['princess_hit_rate']:.4f}"
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


def aggregate_prince(policy: dict) -> dict:
    total = Counter()
    companion = defaultdict(Counter)
    phase = defaultdict(Counter)
    for config in policy["configs"].values():
        prince = config["prince"]
        played = prince["played_events"]
        hand_events = prince["hand_events"]
        total["hand_events"] += hand_events
        total["played_events"] += played
        total["target_self"] += prince["target_self_rate"] * played
        total["target_opponent"] += prince["target_opponent_rate"] * played
        total["princess_hits"] += prince["princess_hit_rate"] * played
        total["self_suicides"] += prince["self_suicide_rate"] * played
        for card, row in prince["companion_rows"].items():
            hand = row["hand_events"]
            pl = row["played_events"]
            companion[card]["hand"] += hand
            companion[card]["hand_wins"] += row["hand_winrate"] * hand
            companion[card]["played"] += pl
            companion[card]["played_wins"] += row["played_winrate"] * pl
            companion[card]["princess_hits"] += row["princess_hit_rate"] * pl
        for key, row in prince["phase_rows"].items():
            hand = row["hand_events"]
            pl = row["played_events"]
            phase[key]["hand"] += hand
            phase[key]["hand_wins"] += row["hand_winrate"] * hand
            phase[key]["played"] += pl
            phase[key]["played_wins"] += row["played_winrate"] * pl
    companion_rows = {}
    for card, row in companion.items():
        companion_rows[card] = {
            "name": CARD_NAMES[int(card)],
            "hand_events": int(row["hand"]),
            "hand_winrate": float(row["hand_wins"] / max(1, row["hand"])),
            "played_events": int(row["played"]),
            "play_rate": float(row["played"] / max(1, row["hand"])),
            "played_winrate": float(row["played_wins"] / max(1, row["played"])),
            "princess_hit_rate": float(row["princess_hits"] / max(1, row["played"])),
        }
    phase_rows = {}
    for key, row in phase.items():
        phase_rows[key] = {
            "hand_events": int(row["hand"]),
            "hand_winrate": float(row["hand_wins"] / max(1, row["hand"])),
            "played_events": int(row["played"]),
            "play_rate": float(row["played"] / max(1, row["hand"])),
            "played_winrate": float(row["played_wins"] / max(1, row["played"])),
        }
    played_total = total["played_events"]
    return {
        "hand_events": int(total["hand_events"]),
        "played_events": int(played_total),
        "play_rate": float(played_total / max(1, total["hand_events"])),
        "target_self_rate": float(total["target_self"] / max(1, played_total)),
        "target_opponent_rate": float(total["target_opponent"] / max(1, played_total)),
        "princess_hit_rate": float(total["princess_hits"] / max(1, played_total)),
        "self_suicide_rate": float(total["self_suicides"] / max(1, played_total)),
        "companion_rows": companion_rows,
        "phase_rows": phase_rows,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    policies = payload["policies"]
    labels = {
        "baseline": "Step3 rapide",
        "specialist": "Step3 + Prince V1",
    }
    lines = [
        "# Step5 Prince - Evaluation Conditionnelle",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "Lecture principale: les colonnes `Prince en main` et `Prince joue` ne gardent que les parties ou le joueur evalue a rencontre cette carte.",
        "",
        "## Synthese",
        "",
        "| Politique | Composite global | Prince en main | Prince joue | Pct joue Prince | Cible soi | Hit Princesse | Suicide soi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "specialist"]:
        policy = policies[name]
        hand = weighted_conditional(policy, "prince_hand")
        played = weighted_conditional(policy, "prince_played")
        prince = aggregate_prince(policy)
        lines.append(
            f"| {labels[name]} | {policy['composite']:.5f} | "
            f"{pct(hand['winrate'])} (n={hand['games']}) | "
            f"{pct(played['winrate'])} (n={played['games']}) | "
            f"{pct(prince['play_rate'])} | "
            f"{pct(prince['target_self_rate'])} | "
            f"{pct(prince['princess_hit_rate'])} | "
            f"{pct(prince['self_suicide_rate'])} |"
        )
    lines.extend(["", "## Par Carte Accompagnante", ""])
    for name in ["baseline", "specialist"]:
        lines.extend(
            [
                f"### {labels[name]}",
                "",
                "| Carte avec Prince | Occurrences | Pct joue Prince | Winrate avec Prince | Winrate si joue | Hit Princesse |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for card, row in sorted(aggregate_prince(policies[name])["companion_rows"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {card} {row['name']} | {row['hand_events']} | "
                f"{pct(row['play_rate'])} | {pct(row['hand_winrate'])} | "
                f"{pct(row['played_winrate'])} | {pct(row['princess_hit_rate'])} |"
            )
        lines.append("")
    lines.extend(["## Par Moment De Partie", ""])
    for name in ["baseline", "specialist"]:
        lines.extend(
            [
                f"### {labels[name]}",
                "",
                "| Phase | Occurrences Prince en main | Pct joue Prince | Winrate avec Prince | Winrate si joue |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for phase_name_key in ["early", "mid", "late"]:
            row = aggregate_prince(policies[name])["phase_rows"].get(phase_name_key, {})
            lines.append(
                f"| {phase_name_key} | {row.get('hand_events', 0)} | "
                f"{pct(row.get('play_rate', 0.0))} | {pct(row.get('hand_winrate', 0.0))} | "
                f"{pct(row.get('played_winrate', 0.0))} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step5 Prince local specialist.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=2400000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--example-limit", type=int, default=24)
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--retarget-margin", type=float, default=0.10)
    parser.add_argument("--veto-score", type=float, default=0.05)
    parser.add_argument("--force-score", type=float, default=0.32)
    parser.add_argument("--self-force-score", type=float, default=0.55)
    parser.add_argument("--min-princess-prob", type=float, default=0.24)
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="prince_specialist_eval.json")
    parser.add_argument("--markdown", default="prince_specialist_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/cards/prince/logs/2026-04-26_prince_specialist_eval.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step5 Prince",
        expected="Comparer Step3 seul et Step3+Prince V1, avec lecture conditionnelle Prince en main.",
        actual=f"games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    context = prepare_policy_context("step3_fast_dagger", args)
    policies = {
        "baseline": evaluate_policy("baseline", "baseline", args, logger, context),
        "specialist": evaluate_policy("specialist", "specialist", args, logger, context),
    }
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "policies": policies,
        "summary": {
            name: {
                "composite": policy["composite"],
                "prince_hand": weighted_conditional(policy, "prince_hand"),
                "prince_played": weighted_conditional(policy, "prince_played"),
                "prince": aggregate_prince(policy),
            }
            for name, policy in policies.items()
        },
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step5 Prince",
        expected="Obtenir une premiere decision conditionnelle sur Prince V1.",
        actual=f"json={output}, markdown={markdown}",
        details=payload["summary"],
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
