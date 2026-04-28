"""Evaluate a trained step3 regret-override head in the arena."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import (
    ExperimentLogger,
    arena_summary,
    composite_score,
    now_stamp,
    resolve_checkpoint,
)
from step2_rl_finetune.evaluate_step2 import (
    OPPONENT_CONFIGS,
    evaluate_player0_model,
    random_action,
    summarize_rewards,
)
from step3_action_value.mini_rollout_probe import classify_state, decode_action
from step3_action_value.train_regret_override import RegretOverrideHead, _debug_belief_array


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_override_checkpoint(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    candidates = [
        path,
        CHECKPOINT_DIR / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Override checkpoint not found: {name_or_path}")


def load_override_bundle(path: str | Path, base_override: str | None = None):
    checkpoint = resolve_override_checkpoint(path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "step3_regret_override_v1":
        raise ValueError(f"{checkpoint} is not a step3 regret override checkpoint")
    head = RegretOverrideHead(hidden_dim=int(ckpt.get("head_hidden_dim", 192)))
    head.load_state_dict(ckpt["head"])
    head.eval()
    base_ref = base_override or ckpt["base_checkpoint"]
    base_checkpoint = resolve_checkpoint(base_ref)
    return checkpoint, base_checkpoint, head, ckpt


class RegretOverridePlayer0:
    def __init__(self, base_checkpoint, head, categories, gate_threshold, device="cpu", example_limit=20):
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.head = head.to(device).eval()
        self.categories = set(categories)
        self.gate_threshold = gate_threshold
        self.device = torch.device(device)
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []
        self.example_limit = example_limit

    def act(self, env, obs_dict):
        base_action, self.state = self.base.act(obs_dict, self.state, agent_id="player_0")
        base_action = int(base_action)
        category = classify_state(env, "player_0")
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1

        if category not in self.categories or int(obs_dict["action_mask"].sum()) <= 1:
            return base_action

        hidden = self.state.detach().cpu().squeeze(0).numpy().astype(np.float32)
        belief = _debug_belief_array(getattr(self.base, "last_debug", None))
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        belief = belief.astype(np.float32)

        with torch.no_grad():
            hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0)
            belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0)
            model_action_t = torch.as_tensor([base_action], dtype=torch.long, device=self.device)
            mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
            logits, gate_logit = self.head(hidden_t, belief_t, model_action_t, mask_t)
            candidate = int(logits.argmax(dim=-1).item())
            gate_prob = float(torch.sigmoid(gate_logit).item())

        self.stats["override_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_gate_prob"] += gate_prob

        if gate_prob >= self.gate_threshold and candidate != base_action:
            self.stats["overrides"] += 1
            self.category_stats[category]["overrides"] += 1
            self.category_stats[category]["sum_override_gate_prob"] += gate_prob
            if len(self.examples) < self.example_limit:
                self.examples.append(
                    {
                        "category": category,
                        "gate_prob": gate_prob,
                        "model_action": base_action,
                        "model_decoded": decode_action(base_action),
                        "chosen_action": candidate,
                        "chosen_decoded": decode_action(candidate),
                    }
                )
            return candidate

        return base_action


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def summarize_override(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["override_checks"])
    return {
        "decisions": int(stats["decisions"]),
        "override_checks": int(stats["override_checks"]),
        "overrides": int(stats["overrides"]),
        "check_rate": float(stats["override_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "mean_gate_prob": float(row["sum_gate_prob"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_gate_prob": float(row["sum_override_gate_prob"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


def evaluate_player0_override(base_checkpoint, head, categories, opponents, games, seed_start, args):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []
    lengths = []
    aggregate_stats = Counter()
    aggregate_categories = defaultdict(Counter)
    examples = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        player0 = RegretOverridePlayer0(
            base_checkpoint,
            head,
            categories,
            gate_threshold=args.gate_threshold,
            device=args.device,
            example_limit=args.example_limit,
        )
        reward0 = 0.0
        actions0 = 0
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                action = player0.act(env, obs_dict)
                actions0 += 1
            else:
                action = opponent_step(env, agent, obs_dict, opponents, bot)
            env.step(action)

        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
        aggregate_stats.update(player0.stats)
        for category, stats in player0.category_stats.items():
            aggregate_categories[category].update(stats)
        if len(examples) < args.example_limit:
            examples.extend(player0.examples[: args.example_limit - len(examples)])

    summary = summarize_rewards(rewards, wins, lengths)
    summary["override"] = summarize_override(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def run_evaluation(base_checkpoint, head, categories, games, seed_start, args, logger=None):
    override_configs = {}
    baseline_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        override_configs[name] = evaluate_player0_override(
            base_checkpoint,
            head,
            categories,
            opponents,
            games,
            seed_start,
            args,
        )
        if args.compare_baseline:
            baseline_configs[name] = evaluate_player0_model(base_checkpoint, opponents, games, seed_start)
        if logger:
            logger.write(
                f"Config {name}",
                expected="La tete d'override doit battre Step2 a seeds identiques.",
                actual=(
                    f"override={override_configs[name]['winrate']:.4f}, "
                    f"overrides={override_configs[name]['override']['overrides']}"
                ),
                details={
                    "override": override_configs[name],
                    "baseline": baseline_configs.get(name),
                },
            )

    report = {
        "created_at": now_stamp(),
        "games": games,
        "seed_start": seed_start,
        "args": vars(args),
        "override_configs": override_configs,
        "override_composite": composite_score(override_configs),
        "baseline_configs": baseline_configs,
        "baseline_composite": composite_score(baseline_configs) if baseline_configs else None,
    }
    if baseline_configs:
        report["override_minus_baseline_composite"] = (
            report["override_composite"] - report["baseline_composite"]
        )
    return report


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate a step3 regret-override checkpoint.")
    parser.add_argument("--checkpoint", default="step3_regret_override_broad_attempt1.pth")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=786000)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="step3_regret_override_eval.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_regret_override_eval.md")
    args = parser.parse_args()

    checkpoint, base_checkpoint, head, ckpt = load_override_bundle(args.checkpoint, args.base_checkpoint)
    categories = args.categories or ckpt.get("categories", [])
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output

    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation regret override",
        expected="Verifier que le correcteur appris bat Step2 sans rollout en direct.",
        actual=f"checkpoint={checkpoint}, base={base_checkpoint}, games={args.games}",
        details={**vars(args), "categories": categories},
    )

    report = run_evaluation(base_checkpoint, head, categories, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin evaluation regret override",
        expected="Verdict succes si composite > Step2 a seeds identiques.",
        actual=f"override_score={report['override_composite']:.5f}",
        details={
            "override": arena_summary(report["override_configs"]),
            "baseline": arena_summary(report["baseline_configs"]) if report["baseline_configs"] else None,
            "delta": report.get("override_minus_baseline_composite"),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
