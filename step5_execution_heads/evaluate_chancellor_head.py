"""Evaluate Step3 fast with a Step5 Chancellor execution head."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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
    TacticalTracker,
    aggregate_outcomes,
    build_roles,
    classify_outcome,
    decode_planned_event,
    make_policy,
    prepare_policy_context,
    role_action,
    summarize_outcomes,
    summarize_tactical,
    tactical_totals,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step5_execution_heads.chancellor_head import load_chancellor_head, score_chancellor_actions


STEP5_DIR = PROJECT_ROOT / "step5_execution_heads"
REPORT_DIR = STEP5_DIR / "reports"
LOG_DIR = STEP5_DIR / "logs"
CHECKPOINT_DIR = STEP5_DIR / "checkpoints"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def resolve_head(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path, CHECKPOINT_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Chancellor head not found: {path}")


def chancellor_candidates(obs_dict: dict) -> list[int]:
    mask = obs_dict["action_mask"]
    return [int(action) for action in np.where(mask == 1)[0] if int(action) >= 900]


class Step3WithChancellorHeadSeat:
    def __init__(self, base_policy, head, args):
        self.base_policy = base_policy
        self.head = head
        self.margin = args.chancellor_margin
        self.device = args.device
        self.stats = Counter()
        self.examples = []
        self.example_limit = args.example_limit

    def act(self, env, obs_dict, agent: str) -> int:
        action = int(self.base_policy.act(env, obs_dict, agent))
        if not (env._chancellor_pending and env.agent_selection == agent):
            return action
        candidates = chancellor_candidates(obs_dict)
        if len(candidates) <= 1:
            self.stats["forced"] += 1
            return action
        if action not in candidates:
            candidates = [action] + candidates
        self.stats["checks"] += 1
        scores = score_chancellor_actions(
            self.head,
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
        self.stats["sum_best_margin"] += best_margin
        if best_action != action and best_margin >= self.margin:
            self.stats["overrides"] += 1
            self.stats[f"override_{action}_to_{best_action}"] += 1
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "pool": list(env._chancellor_pool),
                        "deck_remaining": int(len(env._deck)),
                        "model_action": int(action),
                        "best_action": int(best_action),
                        "margin": best_margin,
                        "scores": {str(a): float(s) for a, s in zip(candidates, centered)},
                    }
                )
            return best_action
        return action


def make_step5_policy(args, roles, eval_agent, context, head):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return Step3WithChancellorHeadSeat(base, head, args)


def evaluate_policy_config(policy_name: str, config_name: str, games: int, seed_start: int, args, context: dict, head=None) -> dict:
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    heuristic_count = CONFIG_HEURISTIC_COUNTS[config_name]
    records = []
    aggregate_tactical = Counter()
    aggregate_step5 = Counter()
    step5_examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        eval_agent = f"player_{game % 4}"
        roles = build_roles(eval_agent, heuristic_count, game)
        if policy_name == "baseline":
            policy = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
        elif policy_name == "step5_chancellor":
            policy = make_step5_policy(args, roles, eval_agent, context, head)
        else:
            raise ValueError(policy_name)
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

            direct_eliminated = []
            if event["kind"] == "card":
                card = event["card"]
                target = event.get("target")
                target_card = event.get("target_card")
                if card == 1 and target and target_card == event.get("guess"):
                    direct_eliminated.append(target)
                elif card == 3 and target:
                    my_after = event.get("remaining_hand", [])
                    my_val = my_after[0] if my_after else None
                    if my_val is not None and target_card is not None:
                        if my_val > target_card:
                            direct_eliminated.append(target)
                        elif target_card > my_val:
                            direct_eliminated.append(agent)
                elif card == 5 and target and target_card == 9:
                    direct_eliminated.append(target)
                elif card == 9:
                    direct_eliminated.append(agent)

            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        if isinstance(policy, Step3WithChancellorHeadSeat):
            aggregate_step5.update(policy.stats)
            step5_examples.extend(policy.examples)
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
        "raw_counts": {key: int(value) for key, value in sorted(aggregate_step5.items())},
        "override_rate": float(aggregate_step5["overrides"] / max(1, aggregate_step5["checks"])),
        "mean_best_margin": float(aggregate_step5["sum_best_margin"] / max(1, aggregate_step5["checks"])),
        "examples": step5_examples[: args.example_limit],
    }
    return summary


def evaluate_policy(policy_name: str, args, logger: ExperimentLogger, context: dict, head=None) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Arena fair seat-rotated Step5 Chancelier.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        result = evaluate_policy_config(policy_name, config_name, args.games, seed_start, args, context, head=head)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter chaque composition terminee.",
            actual=f"winrate={result['winrate']:.4f}, reward={result['mean_reward']:.4f}",
            details={"step5": result.get("step5"), "tactical": result["tactical"]},
        )
    return {"configs": configs, "composite": composite_score(configs)}


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Step5 - Evaluation Tete Chancelier",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Checkpoint: `{payload['head_checkpoint']}`",
        "",
        "## Winrates",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"baseline": "Step3 rapide", "step5_chancellor": "Step3 + tete Chancelier"}
    for name, policy in payload["policies"].items():
        c = policy["configs"]
        lines.append(
            "| {label} | {a} | {b} | {c2} | {d} | {comp:.5f} |".format(
                label=labels[name],
                a=pct(c["vs_0H_3R"]["winrate"]),
                b=pct(c["vs_1H_2R"]["winrate"]),
                c2=pct(c["vs_2H_1R"]["winrate"]),
                d=pct(c["vs_3H"]["winrate"]),
                comp=policy["composite"],
            )
        )
    delta = payload["policies"]["step5_chancellor"]["composite"] - payload["policies"]["baseline"]["composite"]
    lines.extend(["", f"Delta composite: `{delta:+.5f}`.", ""])

    lines.extend(
        [
            "## Tactique Chancelier",
            "",
            "| Politique | Keep highest | Known draw win | Checks | Overrides | Override rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        tact = tactical_totals(policy)
        step5_counts = Counter()
        for config in policy["configs"].values():
            step5_counts.update(config.get("step5", {}).get("raw_counts", {}))
        checks = int(step5_counts.get("checks", 0))
        overrides = int(step5_counts.get("overrides", 0))
        lines.append(
            "| {label} | {keep} | {known} | {checks} | {overrides} | {rate} |".format(
                label=labels[name],
                keep=pct(tact["chancellor_keep_highest_rate"]),
                known=pct(tact["chancellor_known_draw_win_rate"]),
                checks=checks,
                overrides=overrides,
                rate=pct(overrides / max(1, checks)),
            )
        )

    lines.extend(["", "## Sorties Moyennes", "", "| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |", "|---|---:|---:|---:|---:|---:|"])
    for name, policy in payload["policies"].items():
        outcomes = aggregate_outcomes(policy)
        lines.append(
            "| {label} | {winner} | {first} | {second} | {third} | {final} |".format(
                label=labels[name],
                winner=pct(outcomes.get("winner", 0.0)),
                first=pct(outcomes.get("first_out", 0.0)),
                second=pct(outcomes.get("second_out", 0.0)),
                third=pct(outcomes.get("third_out", 0.0)),
                final=pct(outcomes.get("final_loser", 0.0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step5 Chancellor execution head.")
    parser.add_argument("--head", default="chancellor_head_attempt1.pth")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=820000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--chancellor-margin", type=float, default=0.06)
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
    parser.add_argument("--output", default="chancellor_head_attempt1_eval.json")
    parser.add_argument("--markdown", default="chancellor_head_attempt1_eval.md")
    parser.add_argument("--run-log", default="step5_execution_heads/logs/2026-04-26_chancellor_head_attempt1_eval.md")
    args = parser.parse_args()

    head_path = resolve_head(args.head)
    head, ckpt = load_chancellor_head(head_path, args.device)
    context = prepare_policy_context("step3_fast_dagger", args)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut evaluation Step5 Chancelier",
        expected="Comparer Step3 rapide vs Step3 + tete Chancelier sans rollout inference.",
        actual=f"head={head_path}, games={args.games}, margin={args.chancellor_margin}",
        details={"args": vars(args), "head_metadata": ckpt.get("metadata", {})},
    )
    policies = {
        "baseline": evaluate_policy("baseline", args, logger, context),
        "step5_chancellor": evaluate_policy("step5_chancellor", args, logger, context, head=head),
    }
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "head_checkpoint": str(head_path),
        "policies": policies,
    }
    out = REPORT_DIR / args.output
    md = REPORT_DIR / args.markdown
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, md)
    delta = policies["step5_chancellor"]["composite"] - policies["baseline"]["composite"]
    logger.write(
        "Fin evaluation Step5 Chancelier",
        expected="Mesurer si la tete Chancelier ameliore le composite sans rollouts.",
        actual=f"baseline={policies['baseline']['composite']:.5f}, step5={policies['step5_chancellor']['composite']:.5f}, delta={delta:+.5f}",
        details={"report": str(out), "markdown": str(md)},
    )
    print(json.dumps({"baseline": policies["baseline"]["composite"], "step5": policies["step5_chancellor"]["composite"], "delta": delta}, indent=2))


if __name__ == "__main__":
    main()

