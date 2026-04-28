"""Evaluate a Step5 target-execution head against Step3 fast and random target control."""

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
    role_action,
    summarize_outcomes,
    summarize_tactical,
    tactical_totals,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, composite_score, now_stamp
from step5_execution_heads.target_head import (
    KIND_TO_CARD,
    KIND_TO_LABEL,
    action_card,
    action_target,
    infer_kept_card,
    load_target_head,
    score_target_actions,
    target_distribution_from_obs,
)


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
    raise FileNotFoundError(f"Target head not found: {path}")


def valid_actions(mask: np.ndarray) -> list[int]:
    return [int(action) for action in np.where(mask == 1)[0]]


def same_card_candidates(mask: np.ndarray, card: int) -> list[int]:
    return [action for action in valid_actions(mask) if action_card(action) == int(card)]


def is_applicable_target_state(env, agent: str, action: int, kind: str, low_max: int) -> bool:
    event = decode_planned_event(env, agent, int(action))
    if event["kind"] != "card":
        return False
    card = int(event["card"])
    if card != KIND_TO_CARD[kind]:
        return False
    if kind == "baron_low_target":
        remaining = event.get("remaining_hand", [])
        kept = int(remaining[0]) if remaining else infer_kept_card(3, env._hands.get(agent, []))
        return kept <= low_max
    return True


class Step5TargetSeat:
    def __init__(self, base_policy, head, args, mode: str):
        self.base_policy = base_policy
        self.head = head
        self.kind = args.kind
        self.card = KIND_TO_CARD[args.kind]
        self.margin = args.target_margin
        self.low_max = args.baron_low_max
        self.device = args.device
        self.mode = mode
        self.baron_rule_style = args.baron_rule_style
        self.stats = Counter()
        self.examples = []
        self.example_limit = args.example_limit

    def _baron_rule_scores(self, env, obs_dict, agent: str, candidates: list[int]) -> np.ndarray:
        kept = infer_kept_card(3, list(env._hands.get(agent, [])))
        obs = np.asarray(obs_dict["observation"], dtype=np.float32)
        agent_idx = int(str(agent).rsplit("_", 1)[1])
        deck_remaining = int(len(env._deck))
        phase_weight = 1.00 if deck_remaining <= 5 else (0.70 if deck_remaining <= 10 else 0.40)
        scores = []
        for action in candidates:
            target = action_target(action)
            dist = target_distribution_from_obs(obs, agent, target)
            probs = dist["probs"]
            p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
            p_equal = float(probs[kept]) if 0 <= kept < 10 else 0.0
            p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
            expected = float(np.dot(probs, np.arange(10, dtype=np.float32)))

            if self.baron_rule_style == "tactical":
                rel = (int(target) - agent_idx) % 4
                rel_idx = rel - 1 if rel > 0 else 0
                turn_urgency = (4.0 - float(rel)) / 3.0 if rel > 0 else 0.0
                spy_pressure = float(obs[116 + rel_idx]) if 0 <= rel_idx < 3 and len(obs) >= 119 else 0.0
                played_pressure = float(obs[66 + rel_idx]) if 0 <= rel_idx < 3 and len(obs) >= 69 else 0.0
                unchanged = float(dist["unchanged"])
                score = p_lower - 1.65 * p_higher - 0.10 * p_equal
                threat = (
                    0.45 * (expected / 9.0)
                    + 0.25 * turn_urgency
                    + 0.20 * spy_pressure
                    + 0.15 * played_pressure
                    + 0.10 * unchanged
                )
                score += phase_weight * p_lower * threat
                if dist["known_flag"]:
                    score += 0.30 * (p_lower - 1.50 * p_higher - 0.10 * p_equal)
                if kept >= 8:
                    score += 0.06 * (expected / 9.0)
                elif kept <= 5:
                    score -= 0.20 * p_higher
            else:
                reliable = bool(dist["known_flag"] or dist["public_min"] > 0.0)
                known_bonus = 0.12 if dist["known_flag"] else 0.0
                high_card_bonus = 0.03 * max(0, kept - 4)
                # Death is much more expensive than a kill is valuable.
                score = p_lower - 1.45 * p_higher - 0.15 * p_equal
                score += known_bonus * (p_lower - p_higher)
                score += high_card_bonus
                score -= 0.03 * max(0.0, expected - kept)
                if self.baron_rule_style == "known_ev" and not reliable:
                    score = 0.0
            scores.append(float(score))
        return np.asarray(scores, dtype=np.float32)

    def act(self, env, obs_dict, agent: str) -> int:
        action = int(self.base_policy.act(env, obs_dict, agent))
        if env._chancellor_pending:
            return action
        if not is_applicable_target_state(env, agent, action, self.kind, self.low_max):
            return action
        self.stats["plays"] += 1
        candidates = same_card_candidates(obs_dict["action_mask"], self.card)
        if len(candidates) <= 1:
            self.stats["forced"] += 1
            return action
        if action not in candidates:
            candidates = [action] + candidates
        self.stats["checks"] += 1
        if self.mode == "observe":
            return action
        if self.mode == "random":
            new_action = int(np.random.choice(np.asarray(candidates, dtype=np.int64)))
            if new_action != action:
                self.stats["overrides"] += 1
                self.stats[f"override_{action}_to_{new_action}"] += 1
                if len(self.examples) < self.example_limit:
                    self.examples.append(
                        {
                            "mode": "random",
                            "hand": list(env._hands.get(agent, [])),
                            "deck_remaining": int(len(env._deck)),
                            "model_action": int(action),
                            "new_action": int(new_action),
                            "model_target": action_target(action),
                            "new_target": action_target(new_action),
                        }
                )
            return new_action

        if self.mode == "rule":
            scores = self._baron_rule_scores(env, obs_dict, agent, candidates)
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
                            "mode": "rule",
                            "hand": list(env._hands.get(agent, [])),
                            "deck_remaining": int(len(env._deck)),
                            "model_action": int(action),
                            "best_action": int(best_action),
                            "model_target": action_target(action),
                            "best_target": action_target(best_action),
                            "margin": best_margin,
                            "scores": {str(a): float(s) for a, s in zip(candidates, centered)},
                        }
                    )
                return best_action
            return action

        scores = score_target_actions(
            self.head,
            obs_dict["observation"],
            candidates,
            agent,
            list(env._hands.get(agent, [])),
            len(env._deck),
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
                        "mode": "head",
                        "hand": list(env._hands.get(agent, [])),
                        "deck_remaining": int(len(env._deck)),
                        "model_action": int(action),
                        "best_action": int(best_action),
                        "model_target": action_target(action),
                        "best_target": action_target(best_action),
                        "margin": best_margin,
                        "scores": {str(a): float(s) for a, s in zip(candidates, centered)},
                    }
                )
            return best_action
        return action


def make_target_policy(args, roles, eval_agent, context, head, mode: str):
    base = make_policy("step3_fast_dagger", args, roles, eval_agent, context)
    return Step5TargetSeat(base, head, args, mode)


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
            policy = make_target_policy(args, roles, eval_agent, context, head, mode="observe")
        elif policy_name == "random_target":
            policy = make_target_policy(args, roles, eval_agent, context, head, mode="random")
        elif policy_name == "step5_target":
            policy = make_target_policy(args, roles, eval_agent, context, head, mode=args.step5_mode)
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
            direct_eliminated = direct_eliminations_from_event(event, agent)
            env.step(action)
            for eliminated in direct_eliminated:
                if eliminated not in elimination_order:
                    elimination_order.append(eliminated)
            tracker.observe_known_draw(pre_eval_hand, list(env._hands.get(eval_agent, [])), known_top)

        reward_eval = float(rewards[eval_agent])
        won = int(reward_eval >= 1.0)
        aggregate_tactical.update(tracker.finish_game(env, bool(won)))
        per_game_step5 = {}
        if isinstance(policy, Step5TargetSeat):
            aggregate_step5.update(policy.stats)
            step5_examples.extend(policy.examples)
            per_game_step5 = {key: int(value) for key, value in policy.stats.items() if isinstance(value, (int, np.integer))}
        records.append(
            {
                "seed": seed,
                "seat": eval_agent,
                "reward": reward_eval,
                "won": won,
                "outcome": classify_outcome(eval_agent, reward_eval, elimination_order),
                "elimination_order": elimination_order,
                "roles": roles,
                "card_play_count": int(per_game_step5.get("plays", 0)),
                "randomizable_count": int(per_game_step5.get("checks", 0)),
                "override_count": int(per_game_step5.get("overrides", 0)),
            }
        )

    summary = summarize_outcomes(records)
    summary["conditional"] = summarize_conditionals(records)
    summary["tactical"] = summarize_tactical(aggregate_tactical)
    summary["step5"] = {
        "raw_counts": {key: int(value) for key, value in sorted(aggregate_step5.items())},
        "override_rate": float(aggregate_step5["overrides"] / max(1, aggregate_step5["checks"])),
        "mean_best_margin": float(aggregate_step5["sum_best_margin"] / max(1, aggregate_step5["checks"])),
        "examples": step5_examples[: args.example_limit],
    }
    return summary


def summarize_conditionals(records: list[dict]) -> dict:
    played = [record for record in records if int(record.get("card_play_count", 0)) > 0]
    randomizable = [record for record in records if int(record.get("randomizable_count", 0)) > 0]
    overridden = [record for record in records if int(record.get("override_count", 0)) > 0]
    return {
        "played_card": summarize_outcomes(played) if played else {"games": 0, "winrate": 0.0, "mean_reward": 0.0},
        "randomizable_target": summarize_outcomes(randomizable) if randomizable else {"games": 0, "winrate": 0.0, "mean_reward": 0.0},
        "overridden": summarize_outcomes(overridden) if overridden else {"games": 0, "winrate": 0.0, "mean_reward": 0.0},
    }


def evaluate_policy(policy_name: str, args, logger: ExperimentLogger, context: dict, head=None) -> dict:
    configs = {}
    for idx, config_name in enumerate(CONFIG_HEURISTIC_COUNTS):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected=f"Arena fair seat-rotated Step5 {args.kind}.",
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
    label = KIND_TO_LABEL.get(payload["kind"], payload["kind"])
    lines = [
        f"# Step5 - Evaluation Tete {label}",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Kind: `{payload['kind']}`",
        f"Checkpoint: `{payload['head_checkpoint']}`",
        "",
        "## Winrates",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline": "Step3 rapide",
        "random_target": f"{label} random",
        "step5_target": f"Step3 + {('regle' if payload['args'].get('step5_mode') == 'rule' else 'tete')} {label}",
    }
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
    baseline = payload["policies"]["baseline"]["composite"]
    lines.extend(
        [
            "",
            f"Delta tete vs Step3: `{payload['policies']['step5_target']['composite'] - baseline:+.5f}`.",
            f"Delta random vs Step3: `{payload['policies']['random_target']['composite'] - baseline:+.5f}`.",
            "",
            "## Winrates Conditionnels - Carte Jouee",
            "",
            "Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.",
            "",
            "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        lines.append(
            "| {label} | {a} | {b} | {c2} | {d} | {weighted} |".format(
                label=labels[name],
                a=conditional_cell(policy, "vs_0H_3R", "played_card"),
                b=conditional_cell(policy, "vs_1H_2R", "played_card"),
                c2=conditional_cell(policy, "vs_2H_1R", "played_card"),
                d=conditional_cell(policy, "vs_3H", "played_card"),
                weighted=conditional_weighted_cell(policy, "played_card"),
            )
        )
    lines.extend(
        [
            "",
            "## Winrates Conditionnels - Cible Randomisable",
            "",
            "Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.",
            "",
            "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        lines.append(
            "| {label} | {a} | {b} | {c2} | {d} | {weighted} |".format(
                label=labels[name],
                a=conditional_cell(policy, "vs_0H_3R", "randomizable_target"),
                b=conditional_cell(policy, "vs_1H_2R", "randomizable_target"),
                c2=conditional_cell(policy, "vs_2H_1R", "randomizable_target"),
                d=conditional_cell(policy, "vs_3H", "randomizable_target"),
                weighted=conditional_weighted_cell(policy, "randomizable_target"),
            )
        )
    lines.extend(
        [
            "",
            "## Interventions",
            "",
            "| Politique | Checks | Overrides | Override rate | Mean predicted margin |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, policy in payload["policies"].items():
        step5_counts = Counter()
        mean_num = 0.0
        checks = 0
        for config in policy["configs"].values():
            raw = config.get("step5", {}).get("raw_counts", {})
            step5_counts.update(raw)
            mean_num += float(raw.get("sum_best_margin", 0.0))
            checks += int(raw.get("checks", 0))
        lines.append(
            "| {label} | {checks} | {overrides} | {rate} | {margin:.4f} |".format(
                label=labels[name],
                checks=checks,
                overrides=int(step5_counts.get("overrides", 0)),
                rate=pct(step5_counts.get("overrides", 0) / max(1, checks)),
                margin=mean_num / max(1, checks),
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


def conditional_cell(policy: dict, config_name: str, key: str) -> str:
    row = policy["configs"][config_name].get("conditional", {}).get(key, {})
    games = int(row.get("games", 0))
    if games <= 0:
        return "n/a"
    return f"{pct(row.get('winrate', 0.0))} (n={games})"


def conditional_weighted(policy: dict, key: str) -> dict:
    games = 0
    wins = 0.0
    rewards = 0.0
    for config in policy["configs"].values():
        row = config.get("conditional", {}).get(key, {})
        n = int(row.get("games", 0))
        games += n
        wins += float(row.get("winrate", 0.0)) * n
        rewards += float(row.get("mean_reward", 0.0)) * n
    return {
        "games": int(games),
        "winrate": float(wins / max(1, games)),
        "mean_reward": float(rewards / max(1, games)),
    }


def conditional_weighted_cell(policy: dict, key: str) -> str:
    row = conditional_weighted(policy, key)
    if row["games"] <= 0:
        return "n/a"
    return f"{pct(row['winrate'])} (n={row['games']})"


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step5 target execution head.")
    parser.add_argument("--kind", required=True, choices=list(KIND_TO_CARD))
    parser.add_argument("--head", required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=930000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--target-margin", type=float, default=0.10)
    parser.add_argument("--step5-mode", choices=["head", "rule"], default="head")
    parser.add_argument(
        "--baron-rule-style",
        choices=["ev", "known_ev", "tactical"],
        default="ev",
        help="Rule variant used with --step5-mode rule and --kind baron_target.",
    )
    parser.add_argument("--baron-low-max", type=int, default=4)
    parser.add_argument("--example-limit", type=int, default=16)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--run-log", required=True)
    args = parser.parse_args()

    head_path = resolve_head(args.head)
    head = None
    ckpt = {}
    if args.step5_mode == "head":
        head, ckpt = load_target_head(head_path, args.device)
        if ckpt.get("metadata", {}).get("kind") != args.kind:
            raise ValueError(f"Checkpoint kind {ckpt.get('metadata', {}).get('kind')} does not match --kind {args.kind}")
    elif args.kind != "baron_target":
        raise ValueError("--step5-mode rule is currently implemented only for baron_target")
    context = prepare_policy_context("step3_fast_dagger", args)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        f"Debut evaluation Step5 cible {args.kind}",
        expected="Comparer Step3 rapide, random cible, et tete locale sans rollouts.",
        actual=f"head={head_path}, games={args.games}, margin={args.target_margin}",
        details={"args": vars(args), "head_metadata": ckpt.get("metadata", {})},
    )
    policies = {
        "baseline": evaluate_policy("baseline", args, logger, context),
        "random_target": evaluate_policy("random_target", args, logger, context, head=head),
        "step5_target": evaluate_policy("step5_target", args, logger, context, head=head),
    }
    payload = {
        "created_at": now_stamp(),
        "args": vars(args),
        "kind": args.kind,
        "head_checkpoint": str(head_path),
        "policies": policies,
    }
    out = Path(args.output)
    if out.parent == Path("."):
        out = REPORT_DIR / out
    md = Path(args.markdown)
    if md.parent == Path("."):
        md = REPORT_DIR / md
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, md)
    baseline = policies["baseline"]["composite"]
    result = {
        "baseline": baseline,
        "random_target": policies["random_target"]["composite"],
        "step5_target": policies["step5_target"]["composite"],
        "delta_random": policies["random_target"]["composite"] - baseline,
        "delta_step5": policies["step5_target"]["composite"] - baseline,
        "conditional_played_card": {
            name: conditional_weighted(policy, "played_card")
            for name, policy in policies.items()
        },
        "conditional_randomizable_target": {
            name: conditional_weighted(policy, "randomizable_target")
            for name, policy in policies.items()
        },
    }
    logger.write(
        f"Fin evaluation Step5 cible {args.kind}",
        expected="Mesurer si la tete locale bat Step3 et le controle random.",
        actual=json.dumps(result, ensure_ascii=False),
        details={"report": str(out), "markdown": str(md)},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
