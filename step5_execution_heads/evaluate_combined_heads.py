"""Evaluate Step3 with the validated Step5 Chancellor and Baron modules."""

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
    CONFIG_LABELS,
    TacticalTracker,
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
    alternative_action,
    best_baron_action,
    companion_for_baron,
    direct_eliminations_from_event,
    should_play_baron,
)


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


def action_card(action: int) -> int:
    return int(action) // 100


class Step5CombinedSeat:
    def __init__(self, base_policy, chancellor_head, args, use_chancellor: bool, use_baron: bool):
        self.base_policy = base_policy
        self.chancellor_head = chancellor_head
        self.use_chancellor = use_chancellor
        self.use_baron = use_baron
        self.chancellor_margin = args.chancellor_margin
        self.device = args.device
        self.example_limit = args.example_limit
        self.chancellor_stats = Counter()
        self.baron_stats = Counter()
        self.examples = []

    def _maybe_chancellor(self, env, obs_dict, agent: str, base_action: int) -> int:
        if not self.use_chancellor:
            return base_action
        if not (env._chancellor_pending and env.agent_selection == agent):
            return base_action
        candidates = chancellor_candidates(obs_dict)
        if len(candidates) <= 1:
            self.chancellor_stats["forced"] += 1
            return base_action
        if base_action not in candidates:
            candidates = [base_action] + candidates
        self.chancellor_stats["checks"] += 1
        scores = score_chancellor_actions(
            self.chancellor_head,
            obs_dict["observation"],
            list(env._chancellor_pool),
            len(env._deck),
            candidates,
            device=self.device,
        )
        model_idx = candidates.index(base_action)
        centered = scores - scores[model_idx]
        best_idx = int(np.argmax(centered))
        best_action = int(candidates[best_idx])
        best_margin = float(centered[best_idx])
        self.chancellor_stats["sum_best_margin"] += best_margin
        if best_action != base_action and best_margin >= self.chancellor_margin:
            self.chancellor_stats["overrides"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "module": "chancellor",
                        "pool": list(env._chancellor_pool),
                        "base_action": int(base_action),
                        "chosen": int(best_action),
                        "margin": best_margin,
                    }
                )
            return best_action
        return base_action

    def _maybe_baron(self, env, obs_dict, agent: str, base_action: int) -> int:
        if not self.use_baron or env._chancellor_pending:
            return base_action
        hand = [int(card) for card in env._hands.get(agent, [])]
        companion = companion_for_baron(hand)
        if companion is None:
            return base_action
        self.baron_stats["baron_hand_checks"] += 1
        base_played_baron = action_card(base_action) == 3
        if base_played_baron:
            self.baron_stats["base_baron_plays"] += 1
        best_action, best_stats = best_baron_action(env, obs_dict, agent)
        if best_action is None or best_stats is None:
            return base_action
        alt = alternative_action(env, obs_dict, agent, companion)
        play_baron = should_play_baron(companion, best_stats, base_played_baron)
        chosen = int(best_action) if play_baron else (int(alt) if alt is not None else base_action)
        if not base_played_baron and action_card(chosen) == 3 and not best_stats["reliable"]:
            chosen = base_action
        if chosen != base_action:
            self.baron_stats["overrides"] += 1
            if base_played_baron and action_card(chosen) != 3:
                self.baron_stats["baron_to_other"] += 1
            elif not base_played_baron and action_card(chosen) == 3:
                self.baron_stats["other_to_baron"] += 1
            else:
                self.baron_stats["baron_retarget"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "module": "baron",
                        "hand": hand,
                        "companion": companion,
                        "base_action": int(base_action),
                        "chosen": int(chosen),
                        "best_baron": int(best_action),
                        "alt": alt,
                        "stats": best_stats,
                    }
                )
        return int(chosen)

    def act(self, env, obs_dict, agent: str) -> int:
        base_action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return self._maybe_chancellor(env, obs_dict, agent, base_action)
        return self._maybe_baron(env, obs_dict, agent, base_action)


def make_combined_policy(policy_name: str, args, roles: dict[str, str], eval_agent: str, context: dict, chancellor_head):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return Step5CombinedSeat(
        base,
        chancellor_head,
        args,
        use_chancellor=policy_name in {"chancellor", "both"},
        use_baron=policy_name in {"baron", "both"},
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


def evaluate_policy_config(policy_name: str, config_name: str, games: int, seed_start: int, args, context: dict, chancellor_head) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_tactical = Counter()
    chancellor_stats = Counter()
    baron_stats = Counter()
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        policy = make_combined_policy(policy_name, args, roles, eval_agent, context, chancellor_head)
        policies = {eval_agent: policy}
        tracker = TacticalTracker(eval_agent)
        rewards = {agent: 0.0 for agent in env.possible_agents}
        elimination_order = []

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
            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        chancellor_stats.update(policy.chancellor_stats)
        baron_stats.update(policy.baron_stats)
        examples.extend(policy.examples[: max(0, args.example_limit - len(examples))])
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "elimination_order": elimination_order,
                "roles": roles,
            }
        )

    summary = summarize_outcomes(records)
    summary["tactical"] = summarize_tactical(aggregate_tactical)
    summary["step5"] = {
        "chancellor": {
            "raw_counts": {key: int(value) for key, value in sorted(chancellor_stats.items())},
            "override_rate": float(chancellor_stats["overrides"] / max(1, chancellor_stats["checks"])),
            "mean_best_margin": float(chancellor_stats["sum_best_margin"] / max(1, chancellor_stats["checks"])),
        },
        "baron": {
            "raw_counts": {key: int(value) for key, value in sorted(baron_stats.items())},
            "override_rate": float(baron_stats["overrides"] / max(1, baron_stats["baron_hand_checks"])),
        },
        "examples": examples[: args.example_limit],
    }
    return summary


def evaluate_policy(policy_name: str, args, logger: ExperimentLogger, context: dict, chancellor_head) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Arena fair seat-rotated Step5 combine.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, config_name, args.games, seed_start, args, context, chancellor_head)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque composition terminee.",
            actual=f"winrate={result['winrate']:.4f}, reward={result['mean_reward']:.4f}",
            details={"step5": result["step5"], "tactical": result["tactical"]},
        )
    return {"configs": configs, "composite": composite_score(configs)}


def write_markdown(payload: dict, path: Path) -> None:
    labels = {
        "baseline": "Step3 rapide",
        "chancellor": "Step3 + Chancelier V1",
        "baron": "Step3 + Baron V1",
        "both": "Step3 + Chancelier + Baron",
    }
    lines = [
        "# Step5 - Evaluation Combinee Chancelier + Baron",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parties: `{payload['args']['games']}` par composition.",
        f"Chancelier: `{payload['chancellor_head']}`.",
        "",
        "## Winrates",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", "chancellor", "baron", "both"]:
        policy = payload["policies"][name]
        configs = policy["configs"]
        lines.append(
            f"| {labels[name]} | "
            f"{pct(configs['vs_0H_3R']['winrate'])} | "
            f"{pct(configs['vs_1H_2R']['winrate'])} | "
            f"{pct(configs['vs_2H_1R']['winrate'])} | "
            f"{pct(configs['vs_3H']['winrate'])} | "
            f"{policy['composite']:.5f} |"
        )
    baseline = payload["policies"]["baseline"]["composite"]
    lines.extend(["", "## Deltas Vs Base", ""])
    for name in ["chancellor", "baron", "both"]:
        delta = payload["policies"][name]["composite"] - baseline
        lines.append(f"- {labels[name]}: `{delta:+.5f}` composite.")
    lines.extend(["", "## Tactique Agregee", ""])
    lines.append("| Politique | Guard hit | Baron win | Baron loss | Chancellor keep highest |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ["baseline", "chancellor", "baron", "both"]:
        tactical = tactical_totals(payload["policies"][name])
        lines.append(
            f"| {labels[name]} | "
            f"{pct(tactical.get('guard_hit_rate', 0.0))} | "
            f"{pct(tactical.get('baron_win_rate', 0.0))} | "
            f"{pct(tactical.get('baron_loss_rate', 0.0))} | "
            f"{pct(tactical.get('chancellor_keep_highest_rate', 0.0))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step3 with Chancellor and Baron Step5 modules.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=2200000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--chancellor-head", default="step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth")
    parser.add_argument("--chancellor-margin", type=float, default=0.10)
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
    parser.add_argument("--output", default="combined_chancellor_baron_eval.json")
    parser.add_argument("--markdown", default="combined_chancellor_baron_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/logs/2026-04-26_combined_chancellor_baron_eval.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    head_path = resolve_path(args.chancellor_head)
    chancellor_head, _ckpt = load_chancellor_head(head_path, args.device)
    context = prepare_policy_context("step3_fast_dagger", args)
    logger.write(
        "Debut evaluation combinee Chancelier + Baron",
        expected="Comparer Step3 seul, modules individuels et combinaison.",
        actual=f"games={args.games}, head={head_path}",
        details=vars(args),
    )
    policies = {
        "baseline": evaluate_policy("baseline", args, logger, context, chancellor_head),
        "chancellor": evaluate_policy("chancellor", args, logger, context, chancellor_head),
        "baron": evaluate_policy("baron", args, logger, context, chancellor_head),
        "both": evaluate_policy("both", args, logger, context, chancellor_head),
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
        "Fin evaluation combinee Chancelier + Baron",
        expected="Produire winrates et rapport markdown.",
        actual=f"json={output}, markdown={markdown}",
        details={name: policy["composite"] for name, policy in policies.items()},
    )
    print(json.dumps({name: policy["composite"] for name, policy in policies.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
