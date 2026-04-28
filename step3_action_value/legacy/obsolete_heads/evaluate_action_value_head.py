"""Evaluate a learned action-value head as a fast Step3 override policy."""

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
from step3_action_value.train_action_value_head import ActionValueHead
from step3_action_value.train_regret_override import _debug_belief_array


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_value_checkpoint(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    candidates = [path, CHECKPOINT_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Action-value checkpoint not found: {name_or_path}")


def load_value_bundle(path: str | Path, base_override: str | None = None):
    checkpoint = resolve_value_checkpoint(path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "step3_action_value_head_v1":
        raise ValueError(f"{checkpoint} is not a step3 action-value checkpoint")
    head = ActionValueHead(
        hidden_dim=int(ckpt.get("head_hidden_dim", 192)),
        embed_dim=int(ckpt.get("embed_dim", 16)),
    )
    head.load_state_dict(ckpt["head"])
    head.eval()
    base_checkpoint = resolve_checkpoint(base_override or ckpt["base_checkpoint"])
    return checkpoint, base_checkpoint, head, ckpt


class ActionValuePlayer0:
    def __init__(self, base_checkpoint, head, categories, override_margin, device="cpu", example_limit=20):
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.head = head.to(device).eval()
        self.categories = set(categories)
        self.override_margin = override_margin
        self.device = torch.device(device)
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []
        self.example_limit = example_limit

    def act(self, env, obs_dict):
        model_action, self.state = self.base.act(obs_dict, self.state, agent_id="player_0")
        model_action = int(model_action)
        category = classify_state(env, "player_0")
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1
        if category not in self.categories or int(obs_dict["action_mask"].sum()) <= 1:
            return model_action

        valid_actions = np.where(obs_dict["action_mask"] == 1)[0].astype(np.int64)
        if model_action not in valid_actions:
            valid_actions = np.array([model_action, *valid_actions.tolist()], dtype=np.int64)

        hidden = self.state.detach().cpu().squeeze(0).numpy().astype(np.float32)
        belief = _debug_belief_array(getattr(self.base, "last_debug", None))
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        belief = belief.astype(np.float32)

        with torch.no_grad():
            hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0)
            belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            model_t = torch.full_like(actions_t, model_action)
            hidden_batch = hidden_t.expand(len(valid_actions), -1)
            belief_batch = belief_t.expand(len(valid_actions), -1, -1)
            scores = self.head(hidden_batch, belief_batch, actions_t, model_t)
            best_idx = int(torch.argmax(scores).item())
            best_action = int(valid_actions[best_idx])
            best_score = float(scores[best_idx].item())
            model_positions = np.where(valid_actions == model_action)[0]
            model_idx = int(model_positions[0]) if len(model_positions) else 0
            model_score = float(scores[model_idx].item())

        margin = best_score - model_score
        self.stats["value_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += margin

        if best_action != model_action and margin >= self.override_margin:
            self.stats["overrides"] += 1
            self.stats["sum_override_margin"] += margin
            self.category_stats[category]["overrides"] += 1
            self.category_stats[category]["sum_override_margin"] += margin
            if len(self.examples) < self.example_limit:
                top = torch.topk(scores, k=min(5, len(valid_actions))).indices.cpu().numpy().tolist()
                self.examples.append(
                    {
                        "category": category,
                        "margin": margin,
                        "model_score": model_score,
                        "best_score": best_score,
                        "model_action": model_action,
                        "model_decoded": decode_action(model_action),
                        "chosen_action": best_action,
                        "chosen_decoded": decode_action(best_action),
                        "top_predicted": [
                            {
                                "action": int(valid_actions[idx]),
                                "score": float(scores[idx].item()),
                                "decoded": decode_action(int(valid_actions[idx])),
                            }
                            for idx in top
                        ],
                    }
                )
            return best_action
        return model_action


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def summarize_value(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["value_checks"])
    overrides = max(1, stats["overrides"])
    return {
        "decisions": int(stats["decisions"]),
        "value_checks": int(stats["value_checks"]),
        "overrides": int(stats["overrides"]),
        "check_rate": float(stats["value_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "mean_override_margin": float(stats["sum_override_margin"] / overrides) if stats["overrides"] else 0.0,
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "mean_margin": float(row["sum_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_margin": float(row["sum_override_margin"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


def evaluate_player0_value(base_checkpoint, head, categories, opponents, games, seed_start, args):
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
        player0 = ActionValuePlayer0(
            base_checkpoint,
            head,
            categories,
            override_margin=args.override_margin,
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
    summary["value_override"] = summarize_value(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def run_evaluation(base_checkpoint, head, categories, games, seed_start, args, logger=None):
    value_configs = {}
    baseline_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        value_configs[name] = evaluate_player0_value(
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
                expected="La Q-head doit battre Step2 sans rollouts en direct.",
                actual=(
                    f"value={value_configs[name]['winrate']:.4f}, "
                    f"overrides={value_configs[name]['value_override']['overrides']}"
                ),
                details={"value": value_configs[name], "baseline": baseline_configs.get(name)},
            )
    report = {
        "created_at": now_stamp(),
        "games": games,
        "seed_start": seed_start,
        "args": vars(args),
        "value_configs": value_configs,
        "value_composite": composite_score(value_configs),
        "baseline_configs": baseline_configs,
        "baseline_composite": composite_score(baseline_configs) if baseline_configs else None,
    }
    if baseline_configs:
        report["value_minus_baseline_composite"] = report["value_composite"] - report["baseline_composite"]
    return report


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate a step3 action-value head.")
    parser.add_argument("--checkpoint", default="step3_action_value_head_attempt1.pth")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=788000)
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="step3_action_value_head_eval.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_action_value_head_eval.md")
    args = parser.parse_args()

    checkpoint, base_checkpoint, head, ckpt = load_value_bundle(args.checkpoint, args.base_checkpoint)
    categories = args.categories or ckpt.get("categories", [])
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation action-value head",
        expected="Verifier que la value apprise donne un override rapide gagnant.",
        actual=f"checkpoint={checkpoint}, base={base_checkpoint}, games={args.games}",
        details={**vars(args), "categories": categories},
    )

    report = run_evaluation(base_checkpoint, head, categories, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation action-value head",
        expected="Verdict succes si composite > Step2 a seeds identiques.",
        actual=f"value_score={report['value_composite']:.5f}",
        details={
            "value": arena_summary(report["value_configs"]),
            "baseline": arena_summary(report["baseline_configs"]) if report["baseline_configs"] else None,
            "delta": report.get("value_minus_baseline_composite"),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
