"""Quick evaluation of Step3 with Chancellor, Baron and Prince local modules."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import (
    CONFIG_HEURISTIC_COUNTS,
    aggregate_outcomes,
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
from step5_execution_heads.chancellor_head import load_chancellor_head, score_chancellor_actions
from step5_execution_heads.evaluate_chancellor_head import chancellor_candidates
from step5_execution_heads.cards.baron.evaluate_baron_specialist import (
    alternative_action as baron_alternative_action,
    best_baron_action,
    companion_for_baron,
    direct_eliminations_from_event,
    should_play_baron,
)
from step5_execution_heads.cards.prince.evaluate_prince_specialist import (
    alternative_action as prince_alternative_action,
    best_prince_action,
    companion_for_prince,
    prince_action_stats,
    should_force_prince,
)
from step5_execution_heads.target_head import action_card


STEP5_DIR = PROJECT_ROOT / "step5_execution_heads"
REPORT_DIR = STEP5_DIR / "reports"
LOG_DIR = STEP5_DIR / "logs"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path, PROJECT_ROOT / path, STEP5_DIR / "checkpoints" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


class Step5ThreeSeat:
    def __init__(self, base_policy, chancellor_head, args, use_chancellor: bool, use_baron: bool, use_prince: bool):
        self.base_policy = base_policy
        self.chancellor_head = chancellor_head
        self.use_chancellor = use_chancellor
        self.use_baron = use_baron
        self.use_prince = use_prince
        self.chancellor_margin = args.chancellor_margin
        self.device = args.device
        self.example_limit = args.example_limit
        self.args = args
        self.chancellor_stats = Counter()
        self.baron_stats = Counter()
        self.prince_stats = Counter()
        self.examples = []

    def _maybe_chancellor(self, env, obs_dict, agent: str, action: int) -> int:
        if not self.use_chancellor:
            return action
        if not (env._chancellor_pending and env.agent_selection == agent):
            return action
        candidates = chancellor_candidates(obs_dict)
        if len(candidates) <= 1:
            self.chancellor_stats["forced"] += 1
            return action
        if action not in candidates:
            candidates = [action] + candidates
        self.chancellor_stats["checks"] += 1
        scores = score_chancellor_actions(
            self.chancellor_head,
            obs_dict["observation"],
            list(env._chancellor_pool),
            len(env._deck),
            candidates,
            device=self.device,
        )
        model_idx = candidates.index(action)
        centered = scores - scores[model_idx]
        best_idx = int(np.argmax(centered))
        best_action = int(candidates[best_idx])
        best_margin = float(centered[best_idx])
        self.chancellor_stats["sum_best_margin"] += best_margin
        if best_action != action and best_margin >= self.chancellor_margin:
            self.chancellor_stats["overrides"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "module": "chancellor",
                        "pool": list(env._chancellor_pool),
                        "base_action": int(action),
                        "chosen": int(best_action),
                        "margin": best_margin,
                    }
                )
            return best_action
        return action

    def _maybe_baron(self, env, obs_dict, agent: str, action: int) -> int:
        if not self.use_baron or env._chancellor_pending:
            return action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_baron(hand)
        if companion is None:
            return action
        self.baron_stats["baron_hand_checks"] += 1
        base_played_baron = action_card(action) == 3
        if base_played_baron:
            self.baron_stats["base_baron_plays"] += 1
        best_action, best_stats = best_baron_action(env, obs_dict, agent)
        if best_action is None or best_stats is None:
            return action
        alt = baron_alternative_action(env, obs_dict, agent, companion)
        play_baron = should_play_baron(companion, best_stats, base_played_baron)
        chosen = int(best_action) if play_baron else (int(alt) if alt is not None else action)
        if not base_played_baron and action_card(chosen) == 3 and not best_stats["reliable"]:
            chosen = action
        if chosen != action:
            self.baron_stats["overrides"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "module": "baron",
                        "hand": hand,
                        "companion": companion,
                        "base_action": int(action),
                        "chosen": int(chosen),
                        "best_baron": int(best_action),
                        "alt": alt,
                        "stats": best_stats,
                    }
                )
        return int(chosen)

    def _maybe_prince(self, env, obs_dict, agent: str, action: int) -> int:
        if not self.use_prince or env._chancellor_pending:
            return action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_prince(hand)
        if companion is None:
            return action
        self.prince_stats["prince_hand_checks"] += 1
        base_played_prince = action_card(action) == 5
        if base_played_prince:
            self.prince_stats["base_prince_plays"] += 1
        best_action, best_stats = best_prince_action(env, obs_dict, agent, companion)
        if best_action is None or best_stats is None:
            return action

        chosen = int(action)
        current_stats = None
        if base_played_prince:
            current_stats = prince_action_stats(env, obs_dict, agent, action, companion)
            if best_action != action and best_stats["score"] - current_stats["score"] >= self.args.retarget_margin:
                chosen = int(best_action)
            elif best_stats["score"] < self.args.veto_score and companion in {6, 7, 8}:
                alt = prince_alternative_action(env, obs_dict, agent, companion)
                if alt is not None:
                    chosen = int(alt)
        elif should_force_prince(best_stats, companion, self.args):
            chosen = int(best_action)

        if chosen != action:
            self.prince_stats["overrides"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "module": "prince",
                        "hand": hand,
                        "companion": companion,
                        "base_action": int(action),
                        "chosen": int(chosen),
                        "best_prince": int(best_action),
                        "best_stats": best_stats,
                        "current_stats": current_stats,
                    }
                )
        return int(chosen)

    def act(self, env, obs_dict, agent: str) -> int:
        action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return self._maybe_chancellor(env, obs_dict, agent, action)
        action = self._maybe_baron(env, obs_dict, agent, action)
        action = self._maybe_prince(env, obs_dict, agent, action)
        return int(action)


def make_step5_policy(policy_name: str, args, roles: dict[str, str], eval_agent: str, context: dict, chancellor_head):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return Step5ThreeSeat(
        base,
        chancellor_head,
        args,
        use_chancellor=policy_name in {"cb", "cbp"},
        use_baron=policy_name in {"cb", "cbp"},
        use_prince=policy_name == "cbp",
    )


def role_action(env, agent: str, obs_dict, roles: dict[str, str], policies: dict[str, object], bot: HeuristicBot) -> int:
    role = roles[agent]
    if role == "model":
        return int(policies[agent].act(env, obs_dict, agent))
    if role == "heuristic":
        return int(bot.choose_action(env, agent))
    if role == "random":
        return random_action(obs_dict)
    raise ValueError(role)


def update_prince_counters(counter: Counter, event: dict, won: int) -> None:
    counter["prince_hand_games"] += 1
    counter["prince_hand_wins"] += won
    if event["played"]:
        counter["prince_played_games"] += 1
        counter["prince_played_wins"] += won
        counter[f"target_{event['target_kind']}"] += 1
        if event["princess_hit"]:
            counter["princess_hits"] += 1
        if event["self_suicide"]:
            counter["self_suicides"] += 1


def summarize_prince(records: list[dict], aggregate: Counter) -> dict:
    hand = [row for row in records if row["prince_hand"]]
    played = [row for row in records if row["prince_played"]]
    played_events = aggregate["prince_played_games"]
    return {
        "hand_games": len(hand),
        "hand_winrate": float(sum(row["won"] for row in hand) / max(1, len(hand))),
        "played_games": len(played),
        "played_winrate": float(sum(row["won"] for row in played) / max(1, len(played))),
        "play_rate": float(aggregate["prince_played_games"] / max(1, aggregate["prince_hand_games"])),
        "target_self_rate": float(aggregate["target_self"] / max(1, played_events)),
        "princess_hit_rate": float(aggregate["princess_hits"] / max(1, played_events)),
        "self_suicide_rate": float(aggregate["self_suicides"] / max(1, played_events)),
    }


def evaluate_policy_config(policy_name: str, config_name: str, games: int, seed_start: int, args, context: dict, chancellor_head) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_tactical = Counter()
    aggregate_prince = Counter()
    chancellor_stats = Counter()
    baron_stats = Counter()
    prince_stats = Counter()
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        if policy_name == "baseline":
            policy = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
        else:
            policy = make_step5_policy(policy_name, args, roles, eval_agent, context, chancellor_head)
        policies = {eval_agent: policy}
        tracker = TacticalTrackerLite(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []
        prince_events = []

        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            action = role_action(env, agent, obs_dict, roles, policies, bot)
            event = decode_planned_event(env, agent, action)
            pre_eval_hand = list(env._hands.get(eval_agent, []))
            known_top = env._deck_knowledge.get(eval_agent, {}).get(0)
            if agent == eval_agent:
                tracker.before_eval_action(env, event)
                if not env._chancellor_pending:
                    companion = companion_for_prince(pre_eval_hand)
                    if companion is not None:
                        played_prince = bool(event["kind"] == "card" and int(event["card"]) == 5)
                        target_kind = "none"
                        if played_prince:
                            if event.get("target") == eval_agent:
                                target_kind = "self"
                            elif event.get("target"):
                                target_kind = "opponent"
                        prince_events.append(
                            {
                                "played": played_prince,
                                "target_kind": target_kind,
                                "princess_hit": bool(played_prince and event.get("target_card") == 9),
                                "self_suicide": bool(
                                    played_prince and event.get("target") == eval_agent and event.get("target_card") == 9
                                ),
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
        for event in prince_events:
            update_prince_counters(aggregate_prince, event, won)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        if isinstance(policy, Step5ThreeSeat):
            chancellor_stats.update(policy.chancellor_stats)
            baron_stats.update(policy.baron_stats)
            prince_stats.update(policy.prince_stats)
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

    summary = summarize_outcomes(records)
    summary["tactical"] = summarize_tactical(aggregate_tactical)
    summary["prince"] = summarize_prince(records, aggregate_prince)
    summary["step5"] = {
        "chancellor": {
            "raw_counts": {key: int(value) for key, value in sorted(chancellor_stats.items())},
            "override_rate": float(chancellor_stats["overrides"] / max(1, chancellor_stats["checks"])),
        },
        "baron": {
            "raw_counts": {key: int(value) for key, value in sorted(baron_stats.items())},
            "override_rate": float(baron_stats["overrides"] / max(1, baron_stats["baron_hand_checks"])),
        },
        "prince": {
            "raw_counts": {key: int(value) for key, value in sorted(prince_stats.items())},
            "override_rate": float(prince_stats["overrides"] / max(1, prince_stats["prince_hand_checks"])),
        },
        "examples": examples[: args.example_limit],
    }
    return summary


class TacticalTrackerLite:
    """Small local tracker copied from the interlude tracker with the same counters."""

    def __init__(self, eval_agent: str):
        from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import TacticalTracker

        self._tracker = TacticalTracker(eval_agent)

    def before_eval_action(self, env, event: dict) -> None:
        self._tracker.before_eval_action(env, event)

    def observe_known_draw(self, pre_hand: list[int], post_hand: list[int], known_top: int | None) -> None:
        self._tracker.observe_known_draw(pre_hand, post_hand, known_top)

    def finish_game(self, env, won: bool) -> Counter:
        return self._tracker.finish_game(env, won)


def evaluate_policy(policy_name: str, args, logger: ExperimentLogger, context: dict, chancellor_head) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Mini arena fair seat-rotated pour les trois tetes Step5.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, config_name, args.games, seed_start, args, context, chancellor_head)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque composition terminee.",
            actual=f"winrate={result['winrate']:.4f}, prince_hand={result['prince']['hand_winrate']:.4f}",
        )
    return {"configs": configs, "composite": composite_score(configs)}


def weighted_prince(policy: dict, key: str) -> dict:
    games = 0
    wins = 0.0
    for config in policy["configs"].values():
        prince = config["prince"]
        if key == "hand":
            n = int(prince["hand_games"])
            wr = float(prince["hand_winrate"])
        elif key == "played":
            n = int(prince["played_games"])
            wr = float(prince["played_winrate"])
        else:
            raise ValueError(key)
        games += n
        wins += wr * n
    return {"games": games, "winrate": float(wins / max(1, games))}


def aggregate_prince(policy: dict) -> dict:
    total = Counter()
    for config in policy["configs"].values():
        prince = config["prince"]
        hand_events = int(prince["hand_games"])
        played_events = int(prince["played_games"])
        total["hand"] += hand_events
        total["played"] += played_events
        total["target_self"] += prince["target_self_rate"] * played_events
        total["princess_hits"] += prince["princess_hit_rate"] * played_events
        total["self_suicides"] += prince["self_suicide_rate"] * played_events
    return {
        "play_rate": float(total["played"] / max(1, total["hand"])),
        "target_self_rate": float(total["target_self"] / max(1, total["played"])),
        "princess_hit_rate": float(total["princess_hits"] / max(1, total["played"])),
        "self_suicide_rate": float(total["self_suicides"] / max(1, total["played"])),
    }


def write_markdown(payload: dict, path: Path) -> None:
    labels = {
        "baseline": "Step3 rapide",
        "cb": "Step3 + Chancelier + Baron",
        "cbp": "Step3 + Chancelier + Baron + Prince",
    }
    lines = [
        "# Step5 - Mini Evaluation Trois Tetes",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}` par composition.",
        "",
        "Ce run est un thermometre rapide, pas une validation finale.",
        "",
        "## Winrates",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Prince en main |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "cb", "cbp"]:
        policy = payload["policies"][name]
        configs = policy["configs"]
        prince_hand = weighted_prince(policy, "hand")
        lines.append(
            f"| {labels[name]} | "
            f"{pct(configs['vs_0H_3R']['winrate'])} | "
            f"{pct(configs['vs_1H_2R']['winrate'])} | "
            f"{pct(configs['vs_2H_1R']['winrate'])} | "
            f"{pct(configs['vs_3H']['winrate'])} | "
            f"{policy['composite']:.5f} | "
            f"{pct(prince_hand['winrate'])} (n={prince_hand['games']}) |"
        )
    lines.extend(["", "## Tactique", ""])
    lines.append("| Politique | Guard hit | Baron win | Baron loss | Chancellor keep highest | Prince play | Prince hit Princesse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["baseline", "cb", "cbp"]:
        tactical = tactical_totals(payload["policies"][name])
        prince = aggregate_prince(payload["policies"][name])
        lines.append(
            f"| {labels[name]} | "
            f"{pct(tactical.get('guard_hit_rate', 0.0))} | "
            f"{pct(tactical.get('baron_win_rate', 0.0))} | "
            f"{pct(tactical.get('baron_loss_rate', 0.0))} | "
            f"{pct(tactical.get('chancellor_keep_highest_rate', 0.0))} | "
            f"{pct(prince['play_rate'])} | "
            f"{pct(prince['princess_hit_rate'])} |"
        )
    baseline = payload["policies"]["baseline"]["composite"]
    cb = payload["policies"]["cb"]["composite"]
    cbp = payload["policies"]["cbp"]["composite"]
    lines.extend(
        [
            "",
            "## Lecture Courte",
            "",
            f"- Chancelier+Baron vs Step3: `{cb - baseline:+.5f}` composite.",
            f"- Trois tetes vs Step3: `{cbp - baseline:+.5f}` composite.",
            f"- Ajout Prince sur Chancelier+Baron: `{cbp - cb:+.5f}` composite.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Quick eval Step3 with Chancellor, Baron and Prince modules.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=2600000)
    parser.add_argument("--seed-stride", type=int, default=10000)
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
    parser.add_argument("--output", default="combined_three_heads_quick_eval.json")
    parser.add_argument("--markdown", default="combined_three_heads_quick_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/logs/2026-04-26_combined_three_heads_quick_eval.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    head_path = resolve_path(args.chancellor_head)
    chancellor_head, _ckpt = load_chancellor_head(head_path, args.device)
    context = prepare_policy_context("step3_fast_dagger", args)
    logger.write(
        "Debut mini evaluation trois tetes",
        expected="Comparer Step3, Chancelier+Baron et Chancelier+Baron+Prince.",
        actual=f"games={args.games}, seed_start={args.seed_start}",
        details=vars(args),
    )
    policies = {
        "baseline": evaluate_policy("baseline", args, logger, context, chancellor_head),
        "cb": evaluate_policy("cb", args, logger, context, chancellor_head),
        "cbp": evaluate_policy("cbp", args, logger, context, chancellor_head),
    }
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "chancellor_head": str(head_path),
        "policies": policies,
        "outcomes": {name: aggregate_outcomes(policy) for name, policy in policies.items()},
        "tactical": {name: tactical_totals(policy) for name, policy in policies.items()},
    }
    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin mini evaluation trois tetes",
        expected="Produire un thermometre rapide.",
        actual=f"json={output}, markdown={markdown}",
        details={name: policy["composite"] for name, policy in policies.items()},
    )
    print(json.dumps({name: policy["composite"] for name, policy in policies.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
