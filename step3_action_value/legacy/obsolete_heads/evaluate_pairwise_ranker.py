"""Evaluate a fast Step3 pairwise action-value ranker."""

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
from step2_rl_finetune.common import ExperimentLogger, arena_summary, composite_score, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS, evaluate_player0_model, random_action, summarize_rewards
from step3_action_value.mini_rollout_probe import classify_state, decode_action
from step3_action_value.train_pairwise_ranker import PairwiseActionRanker, candidate_actions
from step3_action_value.train_regret_override import _debug_belief_array


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_ranker_checkpoint(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    candidates = [path, CHECKPOINT_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Step3 ranker checkpoint not found: {name_or_path}")


def load_ranker_bundle(path: str | Path, base_override: str | None = None):
    checkpoint = resolve_ranker_checkpoint(path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "step3_pairwise_action_ranker_v1":
        raise ValueError(f"{checkpoint} is not a Step3 pairwise ranker checkpoint")
    ranker = PairwiseActionRanker(
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        embed_dim=int(ckpt.get("embed_dim", 24)),
    )
    ranker.load_state_dict(ckpt["ranker"])
    ranker.eval()
    base_checkpoint = resolve_checkpoint(base_override or ckpt["base_checkpoint"])
    return checkpoint, base_checkpoint, ranker, ckpt


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


class PairwiseRankerPlayer0:
    def __init__(
        self,
        base_checkpoint,
        ranker,
        categories,
        max_actions,
        override_margin,
        min_deck_progress=0.0,
        min_played_cards=0,
        max_active_players=4,
        device="cpu",
        example_limit=20,
    ):
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.ranker = ranker.to(device).eval()
        self.categories = set(categories)
        self.max_actions = max_actions
        self.override_margin = override_margin
        self.min_deck_progress = min_deck_progress
        self.min_played_cards = min_played_cards
        self.max_active_players = max_active_players
        self.device = torch.device(device)
        self.bot = HeuristicBot()
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []
        self.example_limit = example_limit

    def _score(self, obs, hidden, belief, actions, model_action, heuristic_action):
        n = len(actions)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1, -1)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        model_t = torch.full_like(actions_t, int(model_action))
        heuristic_t = torch.full_like(actions_t, int(heuristic_action))
        with torch.no_grad():
            return self.ranker(obs_t, hidden_t, belief_t, actions_t, model_t, heuristic_t)

    def act(self, env, obs_dict):
        model_action, self.state = self.base.act(obs_dict, self.state, agent_id="player_0")
        model_action = int(model_action)
        category = classify_state(env, "player_0")
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1
        if category not in self.categories or int(obs_dict["action_mask"].sum()) <= 1:
            return model_action

        deck_progress = 1.0 - (len(env._deck) / 21.0)
        played_cards = sum(len(cards) for cards in env._played_cards.values())
        active_players = sum(1 for agent in env.possible_agents if not env.terminations.get(agent, False))
        if (
            deck_progress < self.min_deck_progress
            or played_cards < self.min_played_cards
            or active_players > self.max_active_players
        ):
            self.stats["stage_skips"] += 1
            self.category_stats[category]["stage_skips"] += 1
            return model_action

        heuristic_action = int(self.bot.choose_action(env, "player_0"))
        actions = candidate_actions(env, model_action, heuristic_action, self.max_actions)
        if model_action not in actions:
            actions = [model_action] + actions[: self.max_actions - 1]

        hidden = self.state.detach().cpu().squeeze(0).numpy().astype(np.float32)
        belief = _debug_belief_array(getattr(self.base, "last_debug", None))
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        belief = belief.astype(np.float32)
        scores = self._score(
            obs_dict["observation"].astype(np.float32),
            hidden,
            belief,
            actions,
            model_action,
            heuristic_action,
        )
        best_idx = int(torch.argmax(scores).item())
        best_action = int(actions[best_idx])
        best_score = float(scores[best_idx].item())
        model_idx = actions.index(model_action)
        model_score = float(scores[model_idx].item())
        margin = best_score - model_score

        self.stats["ranker_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += margin

        if best_action != model_action and margin >= self.override_margin:
            self.stats["overrides"] += 1
            self.stats["sum_override_margin"] += margin
            self.category_stats[category]["overrides"] += 1
            self.category_stats[category]["sum_override_margin"] += margin
            if len(self.examples) < self.example_limit:
                top_idx = torch.topk(scores, k=min(5, len(actions))).indices.cpu().numpy().tolist()
                self.examples.append(
                    {
                        "category": category,
                        "margin": margin,
                        "model_action": model_action,
                        "model_decoded": decode_action(model_action),
                        "heuristic_action": heuristic_action,
                        "heuristic_decoded": decode_action(heuristic_action),
                        "chosen_action": best_action,
                        "chosen_decoded": decode_action(best_action),
                        "top_predicted": [
                            {
                                "action": int(actions[i]),
                                "score": float(scores[i].item()),
                                "decoded": decode_action(int(actions[i])),
                            }
                            for i in top_idx
                        ],
                    }
                )
            return best_action
        return model_action


def summarize_ranker(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["ranker_checks"])
    overrides = max(1, stats["overrides"])
    return {
        "decisions": int(stats["decisions"]),
        "ranker_checks": int(stats["ranker_checks"]),
        "overrides": int(stats["overrides"]),
        "check_rate": float(stats["ranker_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "stage_skips": int(stats["stage_skips"]),
        "mean_override_margin": float(stats["sum_override_margin"] / overrides) if stats["overrides"] else 0.0,
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "stage_skips": int(row["stage_skips"]),
                "mean_margin": float(row["sum_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_margin": float(row["sum_override_margin"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


def evaluate_player0_ranker(base_checkpoint, ranker, categories, opponents, games, seed_start, args):
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
        player0 = PairwiseRankerPlayer0(
            base_checkpoint=base_checkpoint,
            ranker=ranker,
            categories=categories,
            max_actions=args.max_actions,
            override_margin=args.override_margin,
            min_deck_progress=args.min_deck_progress,
            min_played_cards=args.min_played_cards,
            max_active_players=args.max_active_players,
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
    summary["ranker"] = summarize_ranker(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def run_evaluation(base_checkpoint, ranker, categories, games, seed_start, args, logger=None):
    ranker_configs = {}
    baseline_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        ranker_configs[name] = evaluate_player0_ranker(
            base_checkpoint,
            ranker,
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
                expected="La Step3 ranker rapide doit battre Step2 a seed identique.",
                actual=(
                    f"ranker={ranker_configs[name]['winrate']:.4f}, "
                    f"overrides={ranker_configs[name]['ranker']['overrides']}"
                ),
                details={"ranker": ranker_configs[name], "baseline": baseline_configs.get(name)},
            )
    report = {
        "created_at": now_stamp(),
        "games": games,
        "seed_start": seed_start,
        "args": vars(args),
        "ranker_configs": ranker_configs,
        "ranker_composite": composite_score(ranker_configs),
        "baseline_configs": baseline_configs,
        "baseline_composite": composite_score(baseline_configs) if baseline_configs else None,
    }
    if baseline_configs:
        report["ranker_minus_baseline_composite"] = report["ranker_composite"] - report["baseline_composite"]
    return report


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate a Step3 pairwise action-value ranker.")
    parser.add_argument("--checkpoint", default="step3_pairwise_ranker_attempt1.pth")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=114000)
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument(
        "--min-deck-progress",
        type=float,
        default=0.0,
        help="Skip ranker overrides before this fraction of the deck has been consumed.",
    )
    parser.add_argument(
        "--min-played-cards",
        type=int,
        default=0,
        help="Skip ranker overrides until at least this many cards have been publicly played.",
    )
    parser.add_argument(
        "--max-active-players",
        type=int,
        default=4,
        help="Skip ranker overrides while more than this many players are still active.",
    )
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="step3_pairwise_ranker_eval.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_pairwise_ranker_eval.md")
    args = parser.parse_args()

    checkpoint, base_checkpoint, ranker, ckpt = load_ranker_bundle(args.checkpoint, args.base_checkpoint)
    categories = args.categories or ckpt.get("categories", [])
    args.max_actions = args.max_actions or int(ckpt.get("max_actions", 14))
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation Step3 pairwise ranker",
        expected="Verifier le candidat Step3 rapide sans rollouts en direct.",
        actual=f"checkpoint={checkpoint}, base={base_checkpoint}, games={args.games}",
        details={**vars(args), "categories": categories},
    )

    report = run_evaluation(base_checkpoint, ranker, categories, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation Step3 pairwise ranker",
        expected="Succes seulement si composite > Step2 sur seeds identiques.",
        actual=(
            f"ranker_score={report['ranker_composite']:.5f}, "
            f"delta={report.get('ranker_minus_baseline_composite')}"
        ),
        details={
            "ranker": arena_summary(report["ranker_configs"]),
            "baseline": arena_summary(report["baseline_configs"]) if report["baseline_configs"] else None,
            "delta": report.get("ranker_minus_baseline_composite"),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
