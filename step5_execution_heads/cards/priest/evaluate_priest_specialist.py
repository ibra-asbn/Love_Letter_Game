"""Evaluate a local Priest target specialist.

Priest is an information card. This first specialist is deliberately scoped:
it only retargets Priest when Step3 already chooses to play Priest. The report
then checks both immediate arena winrate and whether Priest information is later
converted into useful actions, especially Guard.
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
    TacticalTracker,
    build_roles,
    classify_outcome,
    decode_planned_event,
    make_policy,
    prepare_policy_context,
    summarize_outcomes,
    summarize_tactical,
    tactical_totals,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step2_rl_finetune.evaluate_step2 import random_action
from step5_execution_heads.cards.baron.evaluate_baron_specialist import direct_eliminations_from_event
from step5_execution_heads.target_head import action_card, action_target, target_distribution_from_obs


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


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def same_card_actions(mask: np.ndarray, card: int) -> list[int]:
    return [action for action in valid_actions(mask) if action_card(action) == int(card)]


def companion_for_priest(hand: list[int]) -> int | None:
    if 2 not in hand:
        return None
    others = [int(card) for card in hand if int(card) != 2]
    return int(others[0]) if others else 2


def phase_name(deck_remaining: int) -> str:
    if deck_remaining >= 11:
        return "early"
    if deck_remaining >= 6:
        return "mid"
    return "late"


def priest_target_stats(env, obs_dict: dict, agent: str, action: int, companion: int) -> dict:
    target = action_target(action)
    dist = target_distribution_from_obs(obs_dict["observation"], agent, target)
    probs = dist["probs"]
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum() / np.log(10.0))
    expected = float(np.dot(probs, np.arange(10, dtype=np.float32)))
    p_guard = float(probs[1])
    p_guardable = float(1.0 - p_guard)
    p_high = float(probs[7:].sum())
    p_princess = float(probs[9])
    p_countess = float(probs[8])
    p_baron_safe = 0.0
    if companion >= 3:
        p_baron_safe = float(probs[:companion].sum())

    known_penalty = 1.55 if dist["known_flag"] else 0.0
    stale_penalty = 0.20 * float(dist["unchanged"])
    guard_future = 0.0
    if companion == 1:
        # If we keep Guard after Priest, exact future information is valuable,
        # but Guard cannot name Guard.
        guard_future = 1.35 * p_guardable + 0.55 * p_high + 0.30 * entropy
    prince_future = 0.0
    if companion == 5:
        prince_future = 1.05 * p_princess + 0.40 * p_countess + 0.18 * p_high
    baron_future = 0.0
    if companion == 3:
        baron_future = 0.45 * p_baron_safe - 0.25 * float(probs[4:].sum())

    score = (
        0.90 * entropy
        + 0.26 * (expected / 9.0)
        + 0.44 * p_high
        + 0.72 * p_princess
        + 0.28 * p_countess
        + guard_future
        + prince_future
        + baron_future
        + 0.18 * float(dist["public_min"])
        - known_penalty
        - stale_penalty
    )
    return {
        "target": target,
        "score": float(score),
        "entropy": entropy,
        "expected": expected,
        "p_guard": p_guard,
        "p_high": p_high,
        "p_princess": p_princess,
        "p_countess": p_countess,
        "known": bool(dist["known_flag"]),
        "public_min": float(dist["public_min"]),
        "unchanged": float(dist["unchanged"]),
        "companion": companion,
    }


def best_priest_action(env, obs_dict: dict, agent: str, companion: int) -> tuple[int | None, dict | None]:
    candidates = same_card_actions(obs_dict["action_mask"], 2)
    if not candidates:
        return None, None
    scored = [(action, priest_target_stats(env, obs_dict, agent, action, companion)) for action in candidates]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    return int(scored[0][0]), scored[0][1]


def priest_guard_probe_stats(env, obs_dict: dict, agent: str, action: int, companion: int) -> dict:
    target = action_target(action)
    dist = target_distribution_from_obs(obs_dict["observation"], agent, target)
    probs = dist["probs"]
    guess_weights = np.asarray([0.20, 0.0, 0.45, 0.55, 0.65, 0.85, 0.95, 1.15, 1.35, 1.65], dtype=np.float32)
    weighted_guess = probs * guess_weights
    weighted_guess[1] = 0.0
    best_guess = int(np.argmax(weighted_guess))
    best_guess_prob = float(probs[best_guess])
    p_guard = float(probs[1])
    p_high = float(probs[7:].sum())
    p_six_plus = float(probs[6:].sum())
    p_princess = float(probs[9])
    p_countess = float(probs[8])
    p_guardable = float(1.0 - p_guard)
    known_penalty = 4.0 if dist["known_flag"] else 0.0
    guard_in_hand_bonus = 0.42 if companion == 1 else 0.0
    score = (
        1.55 * float(weighted_guess[best_guess])
        + 0.80 * best_guess_prob
        + 0.72 * p_high
        + 0.28 * p_six_plus
        + 0.55 * p_princess
        + 0.26 * p_countess
        + 0.22 * float(dist["public_min"])
        + guard_in_hand_bonus * p_guardable
        - 0.50 * p_guard
        - 0.08 * float(dist["unchanged"])
        - known_penalty
    )
    return {
        "target": target,
        "score": float(score),
        "best_guess": best_guess,
        "best_guess_prob": best_guess_prob,
        "p_guard": p_guard,
        "p_high": p_high,
        "p_princess": p_princess,
        "p_countess": p_countess,
        "known": bool(dist["known_flag"]),
        "public_min": float(dist["public_min"]),
        "unchanged": float(dist["unchanged"]),
        "companion": companion,
    }


def best_priest_guard_probe_action(env, obs_dict: dict, agent: str, companion: int) -> tuple[int | None, dict | None]:
    candidates = same_card_actions(obs_dict["action_mask"], 2)
    if not candidates:
        return None, None
    scored = [(action, priest_guard_probe_stats(env, obs_dict, agent, action, companion)) for action in candidates]
    # This variant is explicitly a "pre-Guard" probe: it avoids already-known
    # cards unless there is literally no unknown legal target.
    unknown = [item for item in scored if not item[1]["known"]]
    selected = unknown if unknown else scored
    selected.sort(key=lambda item: item[1]["score"], reverse=True)
    return int(selected[0][0]), selected[0][1]


class PriestSpecialistSeat:
    def __init__(self, base_policy, mode: str, args):
        self.base_policy = base_policy
        self.mode = mode
        self.args = args
        self.stats = Counter()
        self.examples = []
        self.example_limit = args.example_limit

    def act(self, env, obs_dict, agent: str) -> int:
        base_action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return base_action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_priest(hand)
        if companion is None:
            return base_action

        self.stats["priest_hand_checks"] += 1
        base_played_priest = action_card(base_action) == 2
        if base_played_priest:
            self.stats["base_priest_plays"] += 1
        if self.mode == "baseline" or not base_played_priest:
            return base_action

        candidates = same_card_actions(obs_dict["action_mask"], 2)
        if len(candidates) <= 1:
            return base_action

        if self.mode == "random_target":
            chosen = int(np.random.choice(np.asarray(candidates, dtype=np.int64)))
            if chosen != base_action:
                self.stats["overrides"] += 1
            return chosen

        if self.mode == "guard_probe":
            best_action, best_stats = best_priest_guard_probe_action(env, obs_dict, agent, companion)
            if best_action is None or best_stats is None:
                return base_action
            current_stats = priest_guard_probe_stats(env, obs_dict, agent, base_action, companion)
            chosen = base_action
            if best_action != base_action and best_stats["score"] - current_stats["score"] >= self.args.retarget_margin:
                chosen = int(best_action)
            if chosen != base_action:
                self.stats["overrides"] += 1
                self.stats["guard_probe_retarget"] += 1
                if len(self.examples) < self.example_limit:
                    self.examples.append(
                        {
                            "mode": "guard_probe",
                            "hand": hand,
                            "companion": companion,
                            "base_action": int(base_action),
                            "chosen": int(chosen),
                            "current_stats": current_stats,
                            "best_stats": best_stats,
                        }
                    )
            return int(chosen)

        best_action, best_stats = best_priest_action(env, obs_dict, agent, companion)
        if best_action is None or best_stats is None:
            return base_action
        current_stats = priest_target_stats(env, obs_dict, agent, base_action, companion)
        chosen = base_action
        if best_action != base_action and best_stats["score"] - current_stats["score"] >= self.args.retarget_margin:
            chosen = int(best_action)
        if chosen != base_action:
            self.stats["overrides"] += 1
            self.stats["priest_retarget"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "hand": hand,
                        "companion": companion,
                        "base_action": int(base_action),
                        "chosen": int(chosen),
                        "current_stats": current_stats,
                        "best_stats": best_stats,
                    }
                )
        return int(chosen)


def make_priest_policy(args, roles, eval_agent, context, mode: str):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return PriestSpecialistSeat(base, mode, args)


def update_priest_counters(counter: Counter, event: dict, reward_eval: float, won: int) -> None:
    companion = event["companion"]
    phase = event["phase"]
    counter["priest_hand_events"] += 1
    counter[f"hand_{companion}"] += 1
    counter[f"hand_{companion}_wins"] += won
    counter[f"phase_{phase}"] += 1
    counter[f"phase_{phase}_wins"] += won
    if event["played"]:
        target_card = event["target_card"]
        counter["priest_played_events"] += 1
        counter[f"played_{companion}"] += 1
        counter[f"played_{companion}_wins"] += won
        counter[f"played_{phase}"] += 1
        counter[f"played_{phase}_wins"] += won
        if event["known_before"]:
            counter["priest_on_known"] += 1
        else:
            counter["priest_on_unknown"] += 1
        if target_card is not None:
            counter[f"seen_{target_card}"] += 1
            if target_card >= 7:
                counter["seen_high"] += 1
            if target_card == 9:
                counter["seen_princess"] += 1
            if target_card != 1:
                counter["seen_guardable"] += 1


def summarize_records(records: list[dict], aggregate: Counter, tactical: Counter, policy_stats: Counter, examples: list[dict]) -> dict:
    summary = summarize_outcomes(records)
    priest_hand_records = [row for row in records if row["priest_hand"]]
    priest_played_records = [row for row in records if row["priest_played"]]
    played = aggregate["priest_played_events"]
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
        }
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
        }
    summary["tactical"] = summarize_tactical(tactical)
    summary["priest"] = {
        "hand_games": len(priest_hand_records),
        "hand_winrate": float(sum(row["won"] for row in priest_hand_records) / max(1, len(priest_hand_records))),
        "played_games": len(priest_played_records),
        "played_winrate": float(sum(row["won"] for row in priest_played_records) / max(1, len(priest_played_records))),
        "hand_events": int(aggregate["priest_hand_events"]),
        "played_events": int(played),
        "play_rate": float(played / max(1, aggregate["priest_hand_events"])),
        "on_unknown_rate": float(aggregate["priest_on_unknown"] / max(1, played)),
        "seen_high_rate": float(aggregate["seen_high"] / max(1, played)),
        "seen_princess_rate": float(aggregate["seen_princess"] / max(1, played)),
        "seen_guardable_rate": float(aggregate["seen_guardable"] / max(1, played)),
        "seen_cards": {str(card): int(aggregate[f"seen_{card}"]) for card in range(10)},
        "companion_rows": companion_rows,
        "phase_rows": phase_rows,
    }
    summary["specialist"] = {
        "raw_counts": dict(policy_stats),
        "override_rate": float(policy_stats["overrides"] / max(1, policy_stats["base_priest_plays"])),
        "examples": examples,
    }
    return summary


def evaluate_policy_config(policy_name: str, mode: str, config_name: str, games: int, seed_start: int, args, context: dict) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_priest = Counter()
    aggregate_tactical = Counter()
    policy_stats = Counter()
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policy = make_priest_policy(args, roles, eval_agent, context, mode)
        tracker = TacticalTracker(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        priest_events = []

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
            pre_eval_hand = list(env._hands.get(eval_agent, []))
            known_top = env._deck_knowledge.get(eval_agent, {}).get(0)
            if agent == eval_agent:
                tracker.before_eval_action(env, event)
                if not env._chancellor_pending:
                    companion = companion_for_priest(pre_hand)
                    if companion is not None:
                        played_priest = bool(event["kind"] == "card" and int(event["card"]) == 2)
                        priest_events.append(
                            {
                                "companion": companion,
                                "phase": phase_name(len(env._deck)),
                                "played": played_priest,
                                "target_card": event.get("target_card") if played_priest else None,
                                "known_before": bool(event.get("known_target_card") is not None) if played_priest else False,
                            }
                        )
            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        for event in priest_events:
            update_priest_counters(aggregate_priest, event, reward_eval, won)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        if isinstance(policy, PriestSpecialistSeat):
            policy_stats.update(policy.stats)
            examples.extend(policy.examples[: max(0, args.example_limit - len(examples))])
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "priest_hand": bool(priest_events),
                "priest_played": any(event["played"] for event in priest_events),
            }
        )
    return summarize_records(records, aggregate_priest, aggregate_tactical, policy_stats, examples[: args.example_limit])


def evaluate_policy(policy_name: str, mode: str, args, logger: ExperimentLogger, context: dict) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Evaluer le ciblage Pretre par composition.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, mode, config_name, args.games, seed_start, args, context)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque composition terminee.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"priest_hand={result['priest']['hand_winrate']:.4f}, "
                f"seen_high={result['priest']['seen_high_rate']:.4f}"
            ),
            details={"priest": result["priest"], "tactical": result["tactical"]},
        )
    return {"configs": configs, "composite": composite_score(configs)}


def weighted_priest(policy: dict, key: str) -> dict:
    games = 0
    wins = 0.0
    for config in policy["configs"].values():
        priest = config["priest"]
        if key == "hand":
            n = int(priest["hand_games"])
            wr = float(priest["hand_winrate"])
        elif key == "played":
            n = int(priest["played_games"])
            wr = float(priest["played_winrate"])
        else:
            raise ValueError(key)
        games += n
        wins += wr * n
    return {"games": games, "winrate": float(wins / max(1, games))}


def aggregate_priest(policy: dict) -> dict:
    total = Counter()
    companion = defaultdict(Counter)
    for config in policy["configs"].values():
        priest = config["priest"]
        played = int(priest["played_events"])
        hand = int(priest["hand_events"])
        total["hand"] += hand
        total["played"] += played
        total["on_unknown"] += priest["on_unknown_rate"] * played
        total["seen_high"] += priest["seen_high_rate"] * played
        total["seen_princess"] += priest["seen_princess_rate"] * played
        total["seen_guardable"] += priest["seen_guardable_rate"] * played
        for card, row in priest["companion_rows"].items():
            h = row["hand_events"]
            pl = row["played_events"]
            companion[card]["hand"] += h
            companion[card]["hand_wins"] += row["hand_winrate"] * h
            companion[card]["played"] += pl
            companion[card]["played_wins"] += row["played_winrate"] * pl
    companion_rows = {}
    for card, row in companion.items():
        companion_rows[card] = {
            "name": CARD_NAMES[int(card)],
            "hand_events": int(row["hand"]),
            "hand_winrate": float(row["hand_wins"] / max(1, row["hand"])),
            "played_events": int(row["played"]),
            "play_rate": float(row["played"] / max(1, row["hand"])),
            "played_winrate": float(row["played_wins"] / max(1, row["played"])),
        }
    return {
        "play_rate": float(total["played"] / max(1, total["hand"])),
        "on_unknown_rate": float(total["on_unknown"] / max(1, total["played"])),
        "seen_high_rate": float(total["seen_high"] / max(1, total["played"])),
        "seen_princess_rate": float(total["seen_princess"] / max(1, total["played"])),
        "seen_guardable_rate": float(total["seen_guardable"] / max(1, total["played"])),
        "companion_rows": companion_rows,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    labels = {
        "baseline": "Step3 rapide",
        "random_target": "Pretre target random",
        "guard_probe": "Pretre pre-Garde",
        "specialist": "Step3 + Pretre V1",
    }
    lines = [
        "# Step5 Pretre - Evaluation Ciblage V1",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "Pretre V1 ne decide pas quand jouer Pretre. Il corrige uniquement la cible quand Step3 joue deja Pretre.",
        "",
        "## Synthese",
        "",
        "| Politique | Composite | Pretre en main | Pretre joue | Pct joue Pretre | Pretre sur inconnu | Carte 7+ vue | Princesse vue | Pretre->Garde hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    policy_names = list(payload["policies"].keys())
    for name in policy_names:
        policy = payload["policies"][name]
        hand = weighted_priest(policy, "hand")
        played = weighted_priest(policy, "played")
        priest = aggregate_priest(policy)
        tactical = tactical_totals(policy)
        lines.append(
            f"| {labels[name]} | {policy['composite']:.5f} | "
            f"{pct(hand['winrate'])} (n={hand['games']}) | "
            f"{pct(played['winrate'])} (n={played['games']}) | "
            f"{pct(priest['play_rate'])} | "
            f"{pct(priest['on_unknown_rate'])} | "
            f"{pct(priest['seen_high_rate'])} | "
            f"{pct(priest['seen_princess_rate'])} | "
            f"{pct(tactical.get('priest_to_guard_hit_rate', 0.0))} |"
        )
    lines.extend(["", "## Par Carte Accompagnante", ""])
    for name in [item for item in ["baseline", "guard_probe", "specialist"] if item in payload["policies"]]:
        lines.extend(
            [
                f"### {labels[name]}",
                "",
                "| Carte avec Pretre | Occurrences | Pct joue Pretre | Winrate avec Pretre | Winrate si joue |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for card, row in sorted(aggregate_priest(payload["policies"][name])["companion_rows"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {card} {row['name']} | {row['hand_events']} | "
                f"{pct(row['play_rate'])} | {pct(row['hand_winrate'])} | {pct(row['played_winrate'])} |"
            )
        lines.append("")
    lines.extend(["", "## Lecture Courte", ""])
    baseline = payload["policies"]["baseline"]["composite"]
    random_target = payload["policies"]["random_target"]["composite"]
    specialist = payload["policies"]["specialist"]["composite"]
    lines.extend(
        [
            f"- Pretre target random vs Step3: `{random_target - baseline:+.5f}` composite.",
            f"- Pretre V1 vs Step3: `{specialist - baseline:+.5f}` composite.",
            f"- Pretre V1 vs random target: `{specialist - random_target:+.5f}` composite.",
        ]
    )
    if "guard_probe" in payload["policies"]:
        guard_probe = payload["policies"]["guard_probe"]["composite"]
        lines.append(f"- Pretre pre-Garde vs Step3: `{guard_probe - baseline:+.5f}` composite.")
        lines.append(f"- Pretre pre-Garde vs random target: `{guard_probe - random_target:+.5f}` composite.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step5 Priest target specialist.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=2700000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--retarget-margin", type=float, default=0.12)
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
    parser.add_argument("--output", default="priest_specialist_eval.json")
    parser.add_argument("--markdown", default="priest_specialist_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/cards/priest/logs/2026-04-26_priest_specialist_eval.md")
    parser.add_argument("--include-guard-probe", action="store_true")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step5 Pretre",
        expected="Comparer Step3, cible Pretre random et Pretre V1.",
        actual=f"games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    context = prepare_policy_context("step3_fast_dagger", args)
    policies = {
        "baseline": evaluate_policy("baseline", "baseline", args, logger, context),
        "random_target": evaluate_policy("random_target", "random_target", args, logger, context),
    }
    if args.include_guard_probe:
        policies["guard_probe"] = evaluate_policy("guard_probe", "guard_probe", args, logger, context)
    policies["specialist"] = evaluate_policy("specialist", "specialist", args, logger, context)
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "policies": policies,
        "summary": {
            name: {
                "composite": policy["composite"],
                "priest_hand": weighted_priest(policy, "hand"),
                "priest_played": weighted_priest(policy, "played"),
                "priest": aggregate_priest(policy),
                "tactical": tactical_totals(policy),
            }
            for name, policy in policies.items()
        },
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin Step5 Pretre",
        expected="Produire un premier verdict Pretre V1.",
        actual=f"json={output}, markdown={markdown}",
        details=payload["summary"],
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
