"""Evaluate the Step3 v2 CRN advantage head as a fast override policy."""

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
from step3_action_value.train_advantage_head_v2 import (
    AdvantageHeadV2,
    evaluate_candidate_actions_paired,
    paired_delta_stats,
    state_features,
)
from step3_action_value.common import _debug_belief_array, candidate_actions


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_advantage_checkpoint(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    candidates = [path, CHECKPOINT_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Step3 v2 checkpoint not found: {name_or_path}")


def load_advantage_bundle(path: str | Path, base_override: str | None = None):
    checkpoint = resolve_advantage_checkpoint(path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "step3_advantage_head_v2":
        raise ValueError(f"{checkpoint} is not a Step3 v2 advantage checkpoint")
    head = AdvantageHeadV2(
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        embed_dim=int(ckpt.get("embed_dim", 24)),
        extra_dim=int(ckpt.get("extra_dim", 6)),
    )
    head.load_state_dict(ckpt["head"])
    head.eval()
    base_checkpoint = resolve_checkpoint(base_override or ckpt["base_checkpoint"])
    return checkpoint, base_checkpoint, head, ckpt


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def dynamic_margin(base_margin: float, entropy_scale: float, belief: np.ndarray) -> float:
    # Higher uncertainty means we require a clearer predicted advantage.
    return float(base_margin + entropy_scale * max(0.0, min(1.0, state_belief_entropy(belief))))


def state_belief_entropy(belief: np.ndarray) -> float:
    belief = np.asarray(belief, dtype=np.float32)
    if belief.size == 0:
        return 1.0
    probs = np.clip(belief, 1e-8, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=-1) / np.log(probs.shape[-1])
    return float(np.mean(entropy))


class AdvantageV2Player0:
    def __init__(
        self,
        base_checkpoint,
        head,
        categories,
        max_actions,
        override_margin,
        entropy_margin_scale,
        min_deck_progress=0.0,
        min_played_cards=0,
        max_active_players=4,
        opponents=None,
        verify_rollouts=0,
        verify_min_win_delta=0.0,
        verify_min_score_delta=0.0,
        verify_t_threshold=0.0,
        verify_player0_continuation="heuristic",
        verify_reward_score_weight=0.05,
        heuristic_shuffle_targets=False,
        device="cpu",
        example_limit=20,
    ):
        self.base_checkpoint = base_checkpoint
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.head = head.to(device).eval()
        self.categories = set(categories)
        self.max_actions = max_actions
        self.override_margin = override_margin
        self.entropy_margin_scale = entropy_margin_scale
        self.min_deck_progress = min_deck_progress
        self.min_played_cards = min_played_cards
        self.max_active_players = max_active_players
        self.opponents = opponents
        self.verify_rollouts = verify_rollouts
        self.verify_min_win_delta = verify_min_win_delta
        self.verify_min_score_delta = verify_min_score_delta
        self.verify_t_threshold = verify_t_threshold
        self.verify_player0_continuation = verify_player0_continuation
        self.verify_reward_score_weight = verify_reward_score_weight
        self.heuristic_shuffle_targets = heuristic_shuffle_targets
        self.device = torch.device(device)
        self.bot = HeuristicBot(shuffle_targets=heuristic_shuffle_targets)
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []
        self.example_limit = example_limit

    def _score(self, obs, hidden, belief, extra, actions, model_action, heuristic_action):
        n = len(actions)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1, -1)
        extra_t = torch.as_tensor(extra, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        model_t = torch.full_like(actions_t, int(model_action))
        heuristic_t = torch.full_like(actions_t, int(heuristic_action))
        with torch.no_grad():
            scores = self.head(obs_t, hidden_t, belief_t, extra_t, actions_t, model_t, heuristic_t)
        model_positions = [i for i, action in enumerate(actions) if int(action) == int(model_action)]
        if model_positions:
            scores = scores - scores[model_positions[0]]
        return scores

    def _verify_override(self, env, model_action, best_action, decision_seed):
        if self.verify_rollouts <= 0 or best_action == model_action:
            return True, None

        verify_args = argparse.Namespace(
            rollouts_per_action=self.verify_rollouts,
            player0_continuation=self.verify_player0_continuation,
            reward_score_weight=self.verify_reward_score_weight,
            heuristic_shuffle_targets=self.heuristic_shuffle_targets,
        )
        rows, corr = evaluate_candidate_actions_paired(
            env,
            [int(model_action), int(best_action)],
            self.base_checkpoint,
            self.opponents,
            verify_args,
            decision_seed=decision_seed,
        )
        by_action = {int(row["action"]): row for row in rows}
        model_row = by_action.get(int(model_action))
        best_row = by_action.get(int(best_action))
        if model_row is None or best_row is None:
            return False, {"missing_rows": True, "crn_correlation": corr}
        delta = paired_delta_stats(best_row, model_row)
        accepted = (
            delta["mean_win_delta"] >= self.verify_min_win_delta
            and delta["mean_score_delta"] >= self.verify_min_score_delta
            and delta["t_stat"] >= self.verify_t_threshold
        )
        clean_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
        return accepted, {"delta": delta, "crn_correlation": corr, "rows": clean_rows}

    def act(self, env, obs_dict, decision_seed=0):
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
        extra = state_features(env, belief, obs_dict["action_mask"])
        scores = self._score(
            obs_dict["observation"].astype(np.float32),
            hidden,
            belief,
            extra,
            actions,
            model_action,
            heuristic_action,
        )
        best_idx = int(torch.argmax(scores).item())
        best_action = int(actions[best_idx])
        best_score = float(scores[best_idx].item())
        margin = dynamic_margin(self.override_margin, self.entropy_margin_scale, belief)

        self.stats["advantage_checks"] += 1
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += best_score
        self.category_stats[category]["sum_required_margin"] += margin

        if best_action != model_action and best_score >= margin:
            accepted, verify = self._verify_override(env, model_action, best_action, decision_seed)
            if self.verify_rollouts > 0:
                self.stats["verify_checks"] += 1
                self.category_stats[category]["verify_checks"] += 1
                if verify and "delta" in verify:
                    delta = verify["delta"]
                    self.stats["sum_verify_win_delta"] += delta["mean_win_delta"]
                    self.stats["sum_verify_score_delta"] += delta["mean_score_delta"]
                    self.stats["sum_verify_t"] += delta["t_stat"]
                    self.category_stats[category]["sum_verify_win_delta"] += delta["mean_win_delta"]
                    self.category_stats[category]["sum_verify_score_delta"] += delta["mean_score_delta"]
                    self.category_stats[category]["sum_verify_t"] += delta["t_stat"]
                if accepted:
                    self.stats["verify_accepts"] += 1
                    self.category_stats[category]["verify_accepts"] += 1
                else:
                    self.stats["verify_rejects"] += 1
                    self.category_stats[category]["verify_rejects"] += 1
                    return model_action
            self.stats["overrides"] += 1
            self.stats["sum_override_margin"] += best_score
            self.category_stats[category]["overrides"] += 1
            self.category_stats[category]["sum_override_margin"] += best_score
            if len(self.examples) < self.example_limit:
                top_idx = torch.topk(scores, k=min(5, len(actions))).indices.cpu().numpy().tolist()
                self.examples.append(
                    {
                        "category": category,
                        "predicted_advantage": best_score,
                        "required_margin": margin,
                        "belief_entropy": state_belief_entropy(belief),
                        "model_action": model_action,
                        "model_decoded": decode_action(model_action),
                        "heuristic_action": heuristic_action,
                        "heuristic_decoded": decode_action(heuristic_action),
                        "chosen_action": best_action,
                        "chosen_decoded": decode_action(best_action),
                        "verify": verify,
                        "top_predicted": [
                            {
                                "action": int(actions[i]),
                                "advantage": float(scores[i].item()),
                                "decoded": decode_action(int(actions[i])),
                            }
                            for i in top_idx
                        ],
                    }
                )
            return best_action
        return model_action


def summarize_advantage(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["advantage_checks"])
    overrides = max(1, stats["overrides"])
    return {
        "decisions": int(stats["decisions"]),
        "advantage_checks": int(stats["advantage_checks"]),
        "overrides": int(stats["overrides"]),
        "check_rate": float(stats["advantage_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "stage_skips": int(stats["stage_skips"]),
        "verify_checks": int(stats["verify_checks"]),
        "verify_accepts": int(stats["verify_accepts"]),
        "verify_rejects": int(stats["verify_rejects"]),
        "verify_accept_rate": float(stats["verify_accepts"] / max(1, stats["verify_checks"])),
        "mean_verify_win_delta": float(stats["sum_verify_win_delta"] / max(1, stats["verify_checks"])),
        "mean_verify_score_delta": float(stats["sum_verify_score_delta"] / max(1, stats["verify_checks"])),
        "mean_verify_t": float(stats["sum_verify_t"] / max(1, stats["verify_checks"])),
        "mean_override_margin": float(stats["sum_override_margin"] / overrides) if stats["overrides"] else 0.0,
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "stage_skips": int(row["stage_skips"]),
                "verify_checks": int(row["verify_checks"]),
                "verify_accepts": int(row["verify_accepts"]),
                "verify_rejects": int(row["verify_rejects"]),
                "verify_accept_rate": float(row["verify_accepts"] / max(1, row["verify_checks"])),
                "mean_verify_win_delta": float(row["sum_verify_win_delta"] / max(1, row["verify_checks"])),
                "mean_verify_score_delta": float(row["sum_verify_score_delta"] / max(1, row["verify_checks"])),
                "mean_verify_t": float(row["sum_verify_t"] / max(1, row["verify_checks"])),
                "mean_best_advantage": float(row["sum_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_required_margin": float(row["sum_required_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_margin": float(row["sum_override_margin"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


def evaluate_player0_advantage(base_checkpoint, head, categories, opponents, games, seed_start, args):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=getattr(args, "heuristic_shuffle_targets", False))
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
        player0 = AdvantageV2Player0(
            base_checkpoint=base_checkpoint,
            head=head,
            categories=categories,
            max_actions=args.max_actions,
            override_margin=args.override_margin,
            entropy_margin_scale=args.entropy_margin_scale,
            min_deck_progress=args.min_deck_progress,
            min_played_cards=args.min_played_cards,
            max_active_players=args.max_active_players,
            opponents=opponents,
            verify_rollouts=args.verify_rollouts,
            verify_min_win_delta=args.verify_min_win_delta,
            verify_min_score_delta=args.verify_min_score_delta,
            verify_t_threshold=args.verify_t_threshold,
            verify_player0_continuation=args.verify_player0_continuation,
            verify_reward_score_weight=args.verify_reward_score_weight,
            heuristic_shuffle_targets=getattr(args, "heuristic_shuffle_targets", False),
            device=args.device,
            example_limit=args.example_limit,
        )
        reward0 = 0.0
        actions0 = 0
        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                action = player0.act(env, obs_dict, decision_seed=seed * 100 + turn)
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
    summary["advantage"] = summarize_advantage(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def run_evaluation(base_checkpoint, head, categories, games, seed_start, args, logger=None):
    model_configs = {}
    baseline_configs = {}
    for name, opponents in OPPONENT_CONFIGS.items():
        model_configs[name] = evaluate_player0_advantage(
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
                expected="Step3 v2 doit battre Step2 a seeds identiques ou expliquer ses regressions.",
                actual=(
                    f"step3v2={model_configs[name]['winrate']:.4f}, "
                    f"overrides={model_configs[name]['advantage']['overrides']}"
                ),
                details={"step3_v2": model_configs[name], "baseline": baseline_configs.get(name)},
            )
    report = {
        "created_at": now_stamp(),
        "games": games,
        "seed_start": seed_start,
        "args": vars(args),
        "model_configs": model_configs,
        "model_composite": composite_score(model_configs),
        "baseline_configs": baseline_configs,
        "baseline_composite": composite_score(baseline_configs) if baseline_configs else None,
    }
    if baseline_configs:
        report["model_minus_baseline_composite"] = report["model_composite"] - report["baseline_composite"]
    return report


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step3 v2 CRN advantage head.")
    parser.add_argument("--checkpoint", default="step3_advantage_v2_attempt1.pth")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=132000)
    parser.add_argument("--override-margin", type=float, default=0.08)
    parser.add_argument("--entropy-margin-scale", type=float, default=0.00)
    parser.add_argument("--min-deck-progress", type=float, default=0.0)
    parser.add_argument("--min-played-cards", type=int, default=0)
    parser.add_argument("--max-active-players", type=int, default=4)
    parser.add_argument("--verify-rollouts", type=int, default=0)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.0)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.0)
    parser.add_argument("--verify-t-threshold", type=float, default=0.0)
    parser.add_argument("--verify-player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--verify-reward-score-weight", type=float, default=0.05)
    parser.add_argument("--heuristic-shuffle-targets", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="step3_advantage_v2_eval.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_advantage_v2_eval.md")
    args = parser.parse_args()

    checkpoint, base_checkpoint, head, ckpt = load_advantage_bundle(args.checkpoint, args.base_checkpoint)
    categories = args.categories or ckpt.get("categories", [])
    args.max_actions = args.max_actions or int(ckpt.get("max_actions", 14))
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation Step3 v2 advantage",
        expected="Verifier le nouveau correcteur CRN/advantage contre Step2.",
        actual=f"checkpoint={checkpoint}, base={base_checkpoint}, games={args.games}",
        details={**vars(args), "categories": categories},
    )

    report = run_evaluation(base_checkpoint, head, categories, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin evaluation Step3 v2 advantage",
        expected="Succes si le delta composite est net, stable, et sans regression majeure vs 3H.",
        actual=(
            f"step3v2_score={report['model_composite']:.5f}, "
            f"delta={report.get('model_minus_baseline_composite')}"
        ),
        details={
            "step3_v2": arena_summary(report["model_configs"]),
            "baseline": arena_summary(report["baseline_configs"]) if report["baseline_configs"] else None,
            "delta": report.get("model_minus_baseline_composite"),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
