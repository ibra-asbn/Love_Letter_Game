"""DAgger loop for the Step3 v2 fast advantage head.

Attempt2 learned useful CRN labels on Step2 trajectories, but the pure fast
head drifted once it started taking its own overrides. This script fixes that
distribution shift directly: play the current fast head without verification,
collect the states it creates, label those states offline with the same strict
paired-rollout oracle, aggregate the labels, and fine-tune a new fast head.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
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
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS
from step3_action_value.evaluate_advantage_head_v2 import (
    dynamic_margin,
    load_advantage_bundle,
    state_belief_entropy,
)
from step3_action_value.mini_rollout_probe import classify_state, decode_action
from step3_action_value.train_advantage_head_v2 import (
    CHECKPOINT_DIR,
    LOG_DIR,
    REPORT_DIR,
    AdvantageHeadV2,
    collect_advantage_records,
    evaluate_candidate_actions_paired,
    paired_delta_stats,
    score_candidates,
    state_features,
    train_head,
)
from step3_action_value.common import _debug_belief_array, candidate_actions, opponent_action


DATASET_DIR = PROJECT_ROOT / "step3_action_value" / "datasets"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR, DATASET_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_rows(rows):
    output = []
    for row in rows:
        output.append({key: value for key, value in row.items() if not key.startswith("_")})
    return output


def clone_namespace(args, **overrides):
    data = vars(args).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


class FastAdvantageCollector:
    """Player0 policy used during DAgger collection, with verify disabled."""

    def __init__(
        self,
        base_checkpoint,
        head: AdvantageHeadV2,
        categories,
        max_actions: int,
        override_margin: float,
        entropy_margin_scale: float,
        device: str = "cpu",
    ):
        self.base_checkpoint = base_checkpoint
        self.base = load_belief_policy(base_checkpoint)
        self.state = None
        self.head = head.to(device).eval()
        self.categories = set(categories)
        self.max_actions = int(max_actions)
        self.override_margin = float(override_margin)
        self.entropy_margin_scale = float(entropy_margin_scale)
        self.device = torch.device(device)
        self.bot = HeuristicBot()
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)

    def act_with_features(self, env, obs_dict):
        model_action, self.state = self.base.act(obs_dict, self.state, agent_id="player_0")
        model_action = int(model_action)
        category = classify_state(env, "player_0")
        self.stats["decisions"] += 1
        if category:
            self.category_stats[category]["seen"] += 1

        hidden = self.state.detach().cpu().squeeze(0).numpy().astype(np.float32)
        belief = _debug_belief_array(getattr(self.base, "last_debug", None))
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        belief = belief.astype(np.float32)
        extra = state_features(env, belief, obs_dict["action_mask"])
        heuristic_action = int(self.bot.choose_action(env, "player_0"))
        actions = candidate_actions(env, model_action, heuristic_action, self.max_actions)
        if model_action not in actions:
            actions = [model_action] + actions[: self.max_actions - 1]

        chosen_action = model_action
        best_action = model_action
        best_score = 0.0
        margin = dynamic_margin(self.override_margin, self.entropy_margin_scale, belief)

        if category in self.categories and int(obs_dict["action_mask"].sum()) > 1:
            n = len(actions)
            data = {
                "obs": torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "hidden": torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0),
                "belief": torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0),
                "extra": torch.as_tensor(extra, dtype=torch.float32, device=self.device).unsqueeze(0),
                "actions": torch.as_tensor([actions], dtype=torch.long, device=self.device),
                "model_action": torch.as_tensor([model_action], dtype=torch.long, device=self.device),
                "heuristic_action": torch.as_tensor([heuristic_action], dtype=torch.long, device=self.device),
                "model_index": torch.as_tensor([actions.index(model_action)], dtype=torch.long, device=self.device),
            }
            with torch.no_grad():
                scores = score_candidates(self.head, data, torch.as_tensor([0], device=self.device)).squeeze(0)
            best_idx = int(torch.argmax(scores).item())
            best_action = int(actions[best_idx])
            best_score = float(scores[best_idx].item())
            self.stats["advantage_checks"] += 1
            self.category_stats[category]["checked"] += 1
            self.category_stats[category]["sum_margin"] += best_score
            self.category_stats[category]["sum_required_margin"] += margin
            if best_action != model_action and best_score >= margin:
                chosen_action = best_action
                self.stats["overrides"] += 1
                self.stats["sum_override_margin"] += best_score
                self.category_stats[category]["overrides"] += 1
                self.category_stats[category]["sum_override_margin"] += best_score

        feature = {
            "category": category,
            "obs": obs_dict["observation"].astype(np.float32),
            "hidden": hidden,
            "belief": belief,
            "extra": extra,
            "model_action": int(model_action),
            "heuristic_action": int(heuristic_action),
            "actions": [int(action) for action in actions],
            "chosen_action": int(chosen_action),
            "chosen_by_head": int(best_action),
            "predicted_advantage": float(best_score),
            "required_margin": float(margin),
            "belief_entropy": float(state_belief_entropy(belief)),
        }
        return int(chosen_action), feature


def summarize_policy_stats(policy: FastAdvantageCollector) -> dict:
    decisions = max(1, policy.stats["decisions"])
    checks = max(1, policy.stats["advantage_checks"])
    overrides = max(1, policy.stats["overrides"])
    return {
        "decisions": int(policy.stats["decisions"]),
        "advantage_checks": int(policy.stats["advantage_checks"]),
        "overrides": int(policy.stats["overrides"]),
        "check_rate": float(policy.stats["advantage_checks"] / decisions),
        "override_rate_per_decision": float(policy.stats["overrides"] / decisions),
        "override_rate_per_check": float(policy.stats["overrides"] / checks),
        "mean_override_margin": float(policy.stats["sum_override_margin"] / overrides)
        if policy.stats["overrides"]
        else 0.0,
        "by_category": {
            category: {
                "seen": int(stats["seen"]),
                "checked": int(stats["checked"]),
                "overrides": int(stats["overrides"]),
                "mean_best_advantage": float(stats["sum_margin"] / max(1, stats["checked"])),
                "mean_required_margin": float(stats["sum_required_margin"] / max(1, stats["checked"])),
                "mean_override_margin": float(stats["sum_override_margin"] / max(1, stats["overrides"])),
            }
            for category, stats in sorted(policy.category_stats.items())
        },
    }


def target_reached(pair_counts: Counter, args) -> bool:
    if args.onpolicy_states_per_category_config:
        return all(
            pair_counts[(category, config)] >= args.onpolicy_states_per_category_config
            for category in args.categories
            for config in args.collect_configs
        )
    return False


def can_collect(pair_counts: Counter, category: str, config_name: str, args) -> bool:
    if args.onpolicy_states_per_category_config:
        return pair_counts[(category, config_name)] < args.onpolicy_states_per_category_config
    return True


def label_feature(env, feature: dict, checkpoint: Path, opponents, args, decision_seed: int):
    paired_args = argparse.Namespace(
        rollouts_per_action=args.rollouts_per_action,
        player0_continuation=args.player0_continuation,
        reward_score_weight=args.reward_score_weight,
    )
    rows, corr = evaluate_candidate_actions_paired(
        env,
        feature["actions"],
        checkpoint,
        opponents,
        paired_args,
        decision_seed=decision_seed,
    )
    by_action = {int(row["action"]): row for row in rows}
    model_action = int(feature["model_action"])
    model_row = by_action.get(model_action)
    if model_row is None:
        return None, corr

    actions = []
    targets = []
    label_mask = []
    weights = []
    stats = []
    state_significant = 0
    best_significant_action = model_action
    best_significant_advantage = 0.0
    for row in rows:
        action = int(row["action"])
        delta = paired_delta_stats(row, model_row)
        is_model = action == model_action
        significant = is_model or (
            abs(delta["mean_win_delta"]) >= args.min_win_delta
            and abs(delta["t_stat"]) >= args.t_threshold
        )
        if significant and not is_model:
            state_significant += 1
        target = float(delta["mean_score_delta"])
        if is_model:
            target = 0.0
            weight = 1.0
        elif significant:
            confidence = min(args.max_confidence_weight, max(1.0, abs(delta["t_stat"]) / args.t_threshold))
            effect = min(args.max_confidence_weight, max(1.0, abs(delta["mean_win_delta"]) / args.min_win_delta))
            weight = float(confidence * effect)
        else:
            target = 0.0
            weight = args.tie_weight
        if significant and target > best_significant_advantage:
            best_significant_advantage = target
            best_significant_action = action
        actions.append(action)
        targets.append(target)
        label_mask.append(bool(significant))
        weights.append(float(weight))
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        clean.update(delta)
        clean["significant_vs_model"] = bool(significant and not is_model)
        clean["target_advantage"] = float(target)
        clean["label_weight"] = float(weight)
        stats.append(clean)

    record = {
        "source": "on_policy_dagger",
        "category": feature["category"],
        "config": feature["config"],
        "seed": feature["seed"],
        "turn": feature["turn"],
        "obs": feature["obs"],
        "hidden": feature["hidden"],
        "belief": feature["belief"],
        "extra": feature["extra"],
        "model_action": model_action,
        "heuristic_action": int(feature["heuristic_action"]),
        "actions": actions,
        "targets": targets,
        "label_mask": label_mask,
        "weights": weights,
        "best_significant_action": int(best_significant_action),
        "best_significant_advantage": float(best_significant_advantage),
        "significant_actions": int(state_significant),
        "crn_correlation": corr,
        "deck_size": int(len(env._deck)),
        "active_players": int(sum(1 for player in env.possible_agents if not env.terminations.get(player, False))),
        "played_cards_count": int(sum(len(cards) for cards in env._played_cards.values())),
        "on_policy_chosen_action": int(feature["chosen_action"]),
        "on_policy_chosen_decoded": decode_action(feature["chosen_action"]),
        "head_predicted_action": int(feature["chosen_by_head"]),
        "head_predicted_decoded": decode_action(feature["chosen_by_head"]),
        "head_predicted_advantage": float(feature["predicted_advantage"]),
        "head_required_margin": float(feature["required_margin"]),
        "head_belief_entropy": float(feature["belief_entropy"]),
        "top_actions": stats[: min(8, len(stats))],
    }
    return record, corr


def collect_on_policy_records(args, advantage_checkpoint: Path, logger: ExperimentLogger):
    checkpoint, base_checkpoint, head, ckpt = load_advantage_bundle(advantage_checkpoint)
    categories = args.categories or ckpt.get("categories", [])
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    records = []
    category_counts = Counter()
    config_counts = Counter()
    pair_counts = Counter()
    significant_by_category = Counter()
    positive_override_by_category = Counter()
    crn_correlations = []
    policy_summaries = []

    for game in range(args.onpolicy_collect_games):
        if target_reached(pair_counts, args):
            break
        config_name = args.collect_configs[game % len(args.collect_configs)]
        opponents = OPPONENT_CONFIGS[config_name]
        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        policy = FastAdvantageCollector(
            base_checkpoint=base_checkpoint,
            head=head,
            categories=categories,
            max_actions=args.max_actions,
            override_margin=args.override_margin,
            entropy_margin_scale=args.entropy_margin_scale,
            device=args.device,
        )

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                action, feature = policy.act_with_features(env, obs_dict)
                category = feature["category"]
                if (
                    category in args.categories
                    and can_collect(pair_counts, category, config_name, args)
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    feature.update({"config": config_name, "seed": seed, "turn": turn})
                    record, corr = label_feature(
                        env,
                        feature,
                        base_checkpoint,
                        opponents,
                        args,
                        decision_seed=seed * 100 + turn,
                    )
                    if record is not None:
                        records.append(record)
                        category_counts[category] += 1
                        config_counts[config_name] += 1
                        pair_counts[(category, config_name)] += 1
                        significant_by_category[category] += record["significant_actions"]
                        if (
                            record["best_significant_action"] != record["model_action"]
                            and record["best_significant_advantage"] >= args.min_positive_override
                        ):
                            positive_override_by_category[category] += 1
                    if corr is not None:
                        crn_correlations.append(corr)

                    if len(records) % args.log_every_states == 0:
                        logger.write(
                            "Collecte DAgger on-policy",
                            expected="Capturer les etats crees par la tete rapide sans verify, puis les relabeliser CRN.",
                            actual=(
                                f"states={len(records)}, categories={dict(category_counts)}, "
                                f"positive={dict(positive_override_by_category)}"
                            ),
                            details={
                                "config_counts": dict(config_counts),
                                "pair_counts": {
                                    f"{category}|{config}": count
                                    for (category, config), count in sorted(pair_counts.items())
                                },
                                "significant_by_category": dict(significant_by_category),
                            },
                        )
                env.step(action)
            else:
                env.step(opponent_action(env, agent, obs_dict, opponents, bot))

        policy_summaries.append(summarize_policy_stats(policy))

    if not records:
        raise RuntimeError("No DAgger on-policy records collected")

    non_model_rows = sum(max(0, len(record["actions"]) - 1) for record in records)
    significant_rows = sum(record["significant_actions"] for record in records)
    positive_states = sum(
        1
        for record in records
        if record["best_significant_action"] != record["model_action"]
        and record["best_significant_advantage"] >= args.min_positive_override
    )
    summary = {
        "checkpoint": str(checkpoint),
        "base_checkpoint": str(base_checkpoint),
        "states": len(records),
        "rows": int(sum(len(record["actions"]) for record in records)),
        "category_counts": dict(category_counts),
        "config_counts": dict(config_counts),
        "pair_counts": {
            f"{category}|{config}": count for (category, config), count in sorted(pair_counts.items())
        },
        "significant_action_rows": int(significant_rows),
        "non_model_action_rows": int(non_model_rows),
        "significant_action_rate": float(significant_rows / max(1, non_model_rows)),
        "positive_override_states": int(positive_states),
        "positive_override_state_rate": float(positive_states / len(records)),
        "crn_correlation": {
            "mean": float(np.mean(crn_correlations)) if crn_correlations else None,
            "count": int(len(crn_correlations)),
        },
        "significant_by_category": dict(significant_by_category),
        "positive_override_by_category": dict(positive_override_by_category),
        "policy": {
            "mean_override_rate_per_decision": float(
                np.mean([row["override_rate_per_decision"] for row in policy_summaries])
            ),
            "mean_override_rate_per_check": float(
                np.mean([row["override_rate_per_check"] for row in policy_summaries])
            ),
            "games": len(policy_summaries),
        },
    }
    return records, summary, base_checkpoint


def mark_source(records, source):
    for record in records:
        record.setdefault("source", source)
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Run DAgger for the Step3 v2 fast advantage head.")
    parser.add_argument("--initial-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--output-prefix", default="step3_advantage_v2_dagger")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--include-step2-reference", action="store_true")
    parser.add_argument("--reference-collect-games", type=int, default=12000)
    parser.add_argument("--reference-states-per-category-config", type=int, default=10)
    parser.add_argument("--onpolicy-collect-games", type=int, default=12000)
    parser.add_argument("--onpolicy-states-per-category-config", type=int, default=6)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["guard", "priest", "spy", "king", "prince", "chancellor_card", "baron"],
    )
    parser.add_argument("--collect-configs", nargs="+", default=list(OPPONENT_CONFIGS.keys()))
    parser.add_argument("--rollouts-per-action", type=int, default=24)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--reward-score-weight", type=float, default=0.05)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--min-win-delta", type=float, default=0.125)
    parser.add_argument("--min-positive-override", type=float, default=0.10)
    parser.add_argument("--t-threshold", type=float, default=1.65)
    parser.add_argument("--tie-weight", type=float, default=0.10)
    parser.add_argument("--max-confidence-weight", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--tie-loss-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-min-gap", type=float, default=0.10)
    parser.add_argument("--pairwise-beta", type=float, default=0.10)
    parser.add_argument("--model-zero-weight", type=float, default=0.02)
    parser.add_argument("--trust-region-kl-weight", type=float, default=0.0)
    parser.add_argument("--trust-region-temperature", type=float, default=0.25)
    parser.add_argument("--trust-region-step2-epsilon", type=float, default=0.02)
    parser.add_argument("--trust-region-break-advantage", type=float, default=0.20)
    parser.add_argument("--trust-region-break-weight", type=float, default=0.15)
    parser.add_argument("--eval-margin", type=float, default=0.10)
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--entropy-margin-scale", type=float, default=0.00)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--log-every-states", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=15100)
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_advantage_v2_dagger.md")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut Step3 v2 DAgger",
        expected=(
            "Stabiliser la tete rapide sans verify: collecte on-policy, labels oracle CRN stricts, "
            "aggregation et fine-tuning."
        ),
        actual=f"initial_checkpoint={args.initial_checkpoint}, iterations={args.iterations}",
        details=vars(args),
    )

    current_checkpoint = args.initial_checkpoint
    aggregate_records = []
    all_iteration_reports = []
    base_checkpoint = None

    if args.include_step2_reference:
        ref_args = clone_namespace(
            args,
            start="step2_retarget_distilled_attempt1.pth",
            collect_games=args.reference_collect_games,
            states_per_category=140,
            states_per_category_config=args.reference_states_per_category_config,
            log_every_states=args.log_every_states,
        )
        ref_checkpoint = resolve_checkpoint(ref_args.start)
        reference_records, reference_summary = collect_advantage_records(ref_args, ref_checkpoint, logger)
        aggregate_records.extend(mark_source(reference_records, "step2_reference"))
        base_checkpoint = ref_checkpoint
        logger.write(
            "Reference Step2 ajoutee au dataset DAgger",
            expected="Conserver l'ancrage Step2 pour eviter que la tete rapide oublie sa distribution saine.",
            actual=f"states={reference_summary['states']}, positive={reference_summary['positive_override_states']}",
            details=reference_summary,
        )
    else:
        reference_summary = None

    for iteration in range(1, args.iterations + 1):
        iter_seed = args.seed + iteration * 10000
        iter_args = clone_namespace(args, seed=iter_seed)
        on_policy_records, on_policy_summary, loaded_base_checkpoint = collect_on_policy_records(
            iter_args,
            current_checkpoint,
            logger,
        )
        base_checkpoint = base_checkpoint or loaded_base_checkpoint
        aggregate_records.extend(on_policy_records)

        train_args = clone_namespace(
            args,
            seed=iter_seed + 5000,
            init_head_checkpoint=current_checkpoint,
            output=f"{args.output_prefix}_iter{iteration}.pth",
            report=f"{args.output_prefix}_iter{iteration}_train.json",
        )
        head, history = train_head(aggregate_records, train_args, logger)

        output_path = CHECKPOINT_DIR / f"{args.output_prefix}_iter{iteration}.pth"
        payload = {
            "model_type": "step3_advantage_head_v2",
            "created_at": now_stamp(),
            "base_checkpoint": str(base_checkpoint),
            "head": head.cpu().state_dict(),
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "extra_dim": 6,
            "categories": args.categories,
            "max_actions": args.max_actions,
            "metadata": {
                "dagger": True,
                "iteration": iteration,
                "initial_checkpoint": str(args.initial_checkpoint),
                "previous_checkpoint": str(current_checkpoint),
                "args": vars(args),
                "reference_summary": reference_summary,
                "on_policy_summary": on_policy_summary,
                "aggregate_states": len(aggregate_records),
                "history": history,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

        dataset_path = DATASET_DIR / f"{args.output_prefix}_iter{iteration}_records.pt"
        torch.save(
            {
                "created_at": now_stamp(),
                "iteration": iteration,
                "records": aggregate_records,
                "on_policy_summary": on_policy_summary,
                "reference_summary": reference_summary,
            },
            dataset_path,
        )

        report = {
            "created_at": now_stamp(),
            "iteration": iteration,
            "checkpoint": str(output_path),
            "dataset": str(dataset_path),
            "aggregate_states": len(aggregate_records),
            "reference_summary": reference_summary,
            "on_policy_summary": on_policy_summary,
            "history": history,
            "sample_on_policy_records": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"obs", "hidden", "belief", "extra"}
                }
                for record in on_policy_records[:40]
            ],
        }
        report_path = REPORT_DIR / f"{args.output_prefix}_iter{iteration}_train.json"
        write_json(report_path, report)
        all_iteration_reports.append(report)
        logger.write(
            f"Fin iteration DAgger {iteration}",
            expected="Le nouveau checkpoint doit etre evalue sans verify sur 1000 parties/config.",
            actual=f"checkpoint={output_path}, aggregate_states={len(aggregate_records)}",
            details={
                "on_policy_summary": on_policy_summary,
                "final_metrics": history[-1] if history else None,
            },
        )
        current_checkpoint = output_path

    final_report = {
        "created_at": now_stamp(),
        "initial_checkpoint": str(args.initial_checkpoint),
        "final_checkpoint": str(current_checkpoint),
        "iterations": all_iteration_reports,
    }
    final_path = REPORT_DIR / f"{args.output_prefix}_summary.json"
    write_json(final_path, final_report)
    logger.write(
        "Fin Step3 v2 DAgger",
        expected="Produire une succession de tetes rapides a evaluer sans rollouts d'inference.",
        actual=f"final_checkpoint={current_checkpoint}, summary={final_path}",
        details=final_report,
    )
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
