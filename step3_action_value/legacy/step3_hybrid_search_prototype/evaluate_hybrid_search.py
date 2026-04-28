"""Evaluate an actor + belief + rollout-search Love Letter player.

Step 3 proved that action-value search helps. This step keeps the strong Step2
actor as the default policy, but uses its belief probabilities in two places:

1. to build smarter tactical candidate actions;
2. to determinize hidden opponent hands before rollouts.

The result is intentionally a playable hybrid champion, not a compressed fast
student. A few seconds of thinking is acceptable against humans; stable quality
matters more than raw throughput here.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import absolute_to_relative_mask, relative_to_absolute_action
from step2_rl_finetune.common import ExperimentLogger, arena_summary, composite_score, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import (
    OPPONENT_CONFIGS,
    ModelSeat,
    evaluate_player0_model,
    random_action,
    summarize_rewards,
)
from step3_action_value.evaluate_rollout_guided import evaluate_player0_guided
from step3_action_value.mini_rollout_probe import (
    CARD_NAMES,
    choose_actions_for_probe,
    classify_state,
    decode_action,
)


STEP_DIR = PROJECT_ROOT / "step4_hybrid_champion"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"
NUM_PLAYERS = 4
NUM_CARDS = 10
GUARD_GUESSES = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def extract_belief(policy) -> np.ndarray | None:
    debug = getattr(policy, "last_debug", None)
    if debug is None:
        return None
    if hasattr(debug, "belief_probs") and debug.belief_probs is not None:
        probs = debug.belief_probs
        if hasattr(probs, "detach"):
            return probs.detach().cpu().squeeze(0).numpy()
        return np.asarray(probs).squeeze(0)
    if hasattr(debug, "probs") and debug.probs is not None:
        return np.asarray(debug.probs)
    return None


class BeliefModelSeat:
    """ModelSeat variant exposing the belief used for the latest decision."""

    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None
        self.last_belief = None
        self.last_relative_action = None

    def act(self, obs_dict, agent):
        my_idx = int(agent.rsplit("_", 1)[1])
        relative_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        relative_action, self.state = self.policy.act(relative_obs, self.state, agent_id=agent)
        self.last_relative_action = int(relative_action)
        self.last_belief = extract_belief(self.policy)
        return relative_to_absolute_action(relative_action, my_idx)


def target_belief_dim(my_idx: int, target_idx: int) -> int | None:
    if target_idx >= NUM_PLAYERS or target_idx == my_idx:
        return None
    rel = (target_idx - my_idx) % NUM_PLAYERS
    return rel - 1 if rel else None


def known_card_for(env, observer: str, opp: str) -> int | None:
    known = env._known_cards[observer][opp]
    idx = np.where(known >= 1.0)[0]
    return int(idx[0]) if len(idx) else None


def remove_card(counts: np.ndarray, card: int | None) -> bool:
    if card is None:
        return False
    card = int(card)
    if card < 0 or card >= len(counts) or counts[card] <= 0:
        return False
    counts[card] -= 1
    return True


def draw_weighted(
    counts: np.ndarray,
    rng: np.random.Generator,
    probs: np.ndarray | None = None,
    belief_power: float = 1.25,
    belief_floor: float = 0.02,
) -> int | None:
    available = np.flatnonzero(counts > 0)
    if len(available) == 0:
        return None
    weights = counts[available].astype(np.float64)
    if probs is not None:
        belief = np.asarray(probs, dtype=np.float64)[available]
        belief = np.clip(belief, belief_floor, None)
        weights = weights * np.power(belief, belief_power)
    total = float(weights.sum())
    if total <= 0.0:
        weights = counts[available].astype(np.float64)
        total = float(weights.sum())
    card = int(rng.choice(available, p=weights / total))
    remove_card(counts, card)
    return card


def determinize_for_player_with_belief(base_env, belief_probs, args, observer="player_0", seed=0):
    """Sample hidden state using public counts plus the actor's belief probs."""

    rng = np.random.default_rng(seed)
    env = copy.deepcopy(base_env)
    counts = np.array(env.MAX_CARD_COUNTS, dtype=np.int32)

    for cards in env._played_cards.values():
        for card in cards:
            remove_card(counts, card)

    visible_cards = []
    if env._chancellor_pending and env.agent_selection == observer:
        visible_cards.extend(env._chancellor_pool)
    else:
        visible_cards.extend(env._hands.get(observer, []))
    for card in visible_cards:
        remove_card(counts, card)

    my_idx = env.possible_agents.index(observer)
    for opp in env.possible_agents:
        if opp == observer or env.terminations.get(opp, False) or not env._hands.get(opp):
            continue
        opp_idx = env.possible_agents.index(opp)
        known = known_card_for(env, observer, opp)
        if known is not None and remove_card(counts, known):
            env._hands[opp] = [known]
            continue

        dim = target_belief_dim(my_idx, opp_idx)
        probs = belief_probs[dim] if belief_probs is not None and dim is not None else None
        sampled = draw_weighted(
            counts,
            rng,
            probs=probs,
            belief_power=args.belief_power,
            belief_floor=args.belief_floor,
        )
        env._hands[opp] = [sampled] if sampled is not None else []

    if env._set_aside is not None:
        env._set_aside = draw_weighted(counts, rng)

    deck_len = len(env._deck)
    deck = [None] * deck_len
    observer_knowledge = env._deck_knowledge.get(observer, {})
    for pos, card in observer_knowledge.items():
        idx = deck_len - 1 - int(pos)
        if 0 <= idx < deck_len and deck[idx] is None and remove_card(counts, int(card)):
            deck[idx] = int(card)

    for idx in range(deck_len):
        if deck[idx] is None:
            deck[idx] = draw_weighted(counts, rng)
    env._deck = [int(card) for card in deck if card is not None]
    return env


def opponent_step(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def rollout_once_belief(base_env, first_action, seed, checkpoint, opponents, belief_probs, args):
    env = determinize_for_player_with_belief(base_env, belief_probs, args, "player_0", seed)
    bot = HeuristicBot()
    model = ModelSeat(checkpoint) if args.player0_continuation == "model" else None
    reward0 = 0.0

    obs_dict, reward, terminated, truncated, _info = env.last()
    if env.agent_selection == "player_0":
        reward0 += float(reward)
    if terminated or truncated:
        env.step(None)
    else:
        env.step(int(first_action))

    for agent in env.agent_iter():
        obs_dict, reward, terminated, truncated, _info = env.last()
        if agent == "player_0":
            reward0 += float(reward)
        if terminated or truncated:
            env.step(None)
            continue
        if agent == "player_0":
            if args.player0_continuation == "heuristic":
                action = bot.choose_action(env, agent)
            elif args.player0_continuation == "model":
                action = model.act(obs_dict, agent)
            elif args.player0_continuation == "random":
                action = random_action(obs_dict)
            else:
                raise ValueError(args.player0_continuation)
        else:
            action = opponent_step(env, agent, obs_dict, opponents, bot)
        env.step(action)

    return reward0, int(reward0 >= 1.0)


def evaluate_candidate_actions_belief(env, actions, checkpoint, opponents, belief_probs, args, decision_seed):
    rows = []
    for action in actions:
        rewards = []
        wins = []
        for rollout in range(args.rollouts_per_action):
            seed = decision_seed * 1_000_000 + int(action) * 1000 + rollout
            reward, win = rollout_once_belief(env, int(action), seed, checkpoint, opponents, belief_probs, args)
            rewards.append(reward)
            wins.append(win)
        rows.append(
            {
                "action": int(action),
                "decoded": decode_action(int(action)),
                "winrate": float(np.mean(wins)) if wins else 0.0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "reward_std": float(np.std(rewards)) if rewards else 0.0,
                "wins": int(sum(wins)),
                "rollouts": int(len(wins)),
            }
        )
    rows.sort(key=lambda row: (row["winrate"], row["mean_reward"]), reverse=True)
    return rows


def card_value(card: int | None, princess_penalty=True) -> float:
    if card is None:
        return 0.0
    if princess_penalty and card == 9:
        return -0.5
    return float(card)


def kept_after_play(hand, played_card):
    remaining = list(hand)
    if played_card in remaining:
        remaining.remove(played_card)
    return remaining[0] if remaining else None


def valid_target_indices(env, agent):
    return [
        idx
        for idx, opp in enumerate(env.possible_agents)
        if opp != agent
        and opp in env.agents
        and not env.terminations.get(opp, True)
        and not env._protected.get(opp, False)
    ]


def target_probs(env, agent, target_idx, belief_probs):
    if belief_probs is None:
        return None
    my_idx = env.possible_agents.index(agent)
    dim = target_belief_dim(my_idx, target_idx)
    if dim is None:
        return None
    return np.asarray(belief_probs[dim], dtype=np.float64)


def normalized_entropy(probs):
    if probs is None:
        return 0.0
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
    return float(-(probs * np.log(probs)).sum() / np.log(NUM_CARDS))


def expected_card(probs):
    if probs is None:
        return 4.0
    return float(np.sum(np.asarray(probs, dtype=np.float64) * np.arange(NUM_CARDS)))


def candidate_score(env, agent, action, belief_probs):
    mask = env.observe(agent)["action_mask"]
    if action < 0 or action >= len(mask) or mask[action] != 1:
        return -1e9

    if action >= 900:
        pool = list(env._chancellor_pool)
        pool_size = len(pool)
        if not pool:
            return 0.0
        action_idx = action - 900
        keep_idx = action_idx // 2 if pool_size >= 3 else action_idx
        if keep_idx < 0 or keep_idx >= pool_size:
            keep_idx = 0
        kept = pool[keep_idx]
        # Candidate generation only: rollouts will decide whether keeping the
        # Princess is correct in this exact state.
        return 2.0 + card_value(kept, princess_penalty=True) / 2.0

    card = int(action // 100)
    target_idx = int((action % 100) // 10)
    guess = int(action % 10)
    hand = list(env._hands.get(agent, []))
    kept = kept_after_play(hand, card)
    probs = target_probs(env, agent, target_idx, belief_probs)
    targets_self = target_idx == env.possible_agents.index(agent)

    if card == 1 and probs is not None:
        guard_probs = probs.copy()
        guard_probs[1] = 0.0
        top_guess = int(np.argmax(guard_probs))
        score = 8.0 * float(guard_probs[guess])
        if guess == top_guess:
            score += 2.0
        return score

    if card == 2 and probs is not None:
        target = env.possible_agents[target_idx]
        known = known_card_for(env, agent, target)
        known_bonus = 0.0 if known is not None else 0.75
        return 1.5 + normalized_entropy(probs) + known_bonus + 0.05 * expected_card(probs)

    if card == 3 and probs is not None and kept is not None:
        p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
        p_equal = float(probs[kept])
        p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
        return 5.0 * p_lower - 7.0 * p_higher - 0.5 * p_equal + 0.15 * kept

    if card == 4:
        late_bonus = max(0.0, 6.0 - len(env._deck)) / 6.0
        kept_bonus = 0.15 * card_value(kept, princess_penalty=False)
        return 0.75 + late_bonus + kept_bonus

    if card == 5:
        if targets_self:
            if kept == 9:
                return -8.0
            if kept is not None and kept <= 2:
                return 3.0
            if kept is not None and kept <= 4:
                return 1.0
            return -0.5
        if probs is not None:
            return 8.0 * float(probs[9]) + 3.0 * float(probs[8]) + 1.5 * float(probs[7])
        return 0.5

    if card == 6:
        kept_val = card_value(kept, princess_penalty=True)
        return 2.5 - 0.3 * kept_val

    if card == 7 and probs is not None and kept is not None:
        late = len(env._deck) <= 5
        swap_gain = (expected_card(probs) - kept) / 9.0
        return (4.0 if late else 1.0) * swap_gain - (0.7 if not late else 0.0)

    if card == 0:
        return 0.4 + max(0.0, 5.0 - len(env._deck)) / 10.0

    if card == 8:
        forced = 5 in hand or 7 in hand
        return 4.0 if forced else -2.0

    if card == 9:
        return -10.0

    return 0.0


def belief_top_summary(env, agent, belief_probs):
    if belief_probs is None:
        return None
    my_idx = env.possible_agents.index(agent)
    summary = {}
    for opp_idx, opp in enumerate(env.possible_agents):
        if opp == agent:
            continue
        dim = target_belief_dim(my_idx, opp_idx)
        if dim is None:
            continue
        probs = np.asarray(belief_probs[dim], dtype=np.float64)
        ranked = sorted(
            [
                {"card": int(card), "card_name": CARD_NAMES[int(card)], "prob": float(prob)}
                for card, prob in enumerate(probs)
            ],
            key=lambda item: item["prob"],
            reverse=True,
        )
        summary[opp] = ranked[:3]
    return summary


def choose_hybrid_candidates(env, obs_dict, model_action, heuristic_action, belief_probs, args):
    valid = [int(action) for action in np.where(obs_dict["action_mask"] == 1)[0]]
    selected = []
    reasons = {}

    def add(action, reason):
        action = int(action)
        if action not in valid or action in selected:
            return
        selected.append(action)
        reasons[action] = reason

    add(model_action, "actor_default")
    add(heuristic_action, "heuristic_reference")

    ranked = sorted(
        valid,
        key=lambda action: candidate_score(env, "player_0", int(action), belief_probs),
        reverse=True,
    )
    for action in ranked[: max(args.max_actions, args.belief_candidate_pool)]:
        add(action, "belief_ranked")
        if len(selected) >= args.max_actions:
            break

    if len(selected) < args.max_actions:
        for action in choose_actions_for_probe(env, args.max_actions):
            add(action, "step3_probe_backfill")
            if len(selected) >= args.max_actions:
                break

    return selected[: args.max_actions], {str(action): reasons[action] for action in selected[: args.max_actions]}


def summarize_search(stats, categories):
    decisions = max(1, stats["decisions"])
    checks = max(1, stats["search_checks"])
    overrides = max(1, stats["overrides"])
    return {
        "decisions": int(stats["decisions"]),
        "search_checks": int(stats["search_checks"]),
        "overrides": int(stats["overrides"]),
        "belief_available": int(stats["belief_available"]),
        "check_rate": float(stats["search_checks"] / decisions),
        "override_rate_per_decision": float(stats["overrides"] / decisions),
        "override_rate_per_check": float(stats["overrides"] / checks),
        "mean_candidate_count": float(stats["sum_candidates"] / checks) if stats["search_checks"] else 0.0,
        "mean_override_margin": float(stats["sum_override_margin"] / overrides) if stats["overrides"] else 0.0,
        "by_category": {
            category: {
                "seen": int(row["seen"]),
                "checked": int(row["checked"]),
                "overrides": int(row["overrides"]),
                "mean_check_margin": float(row["sum_margin"] / row["checked"]) if row["checked"] else 0.0,
                "mean_override_margin": float(row["sum_override_margin"] / row["overrides"])
                if row["overrides"]
                else 0.0,
            }
            for category, row in sorted(categories.items())
        },
    }


class HybridSearchPlayer0:
    def __init__(self, checkpoint, opponents, args):
        self.checkpoint = checkpoint
        self.base = BeliefModelSeat(checkpoint)
        self.bot = HeuristicBot()
        self.opponents = opponents
        self.args = args
        self.stats = Counter()
        self.category_stats = defaultdict(Counter)
        self.examples = []

    def act(self, env, obs_dict, decision_seed):
        model_action = int(self.base.act(obs_dict, "player_0"))
        belief_probs = self.base.last_belief
        heuristic_action = int(self.bot.choose_action(env, "player_0"))
        category = classify_state(env, "player_0")

        self.stats["decisions"] += 1
        if belief_probs is not None:
            self.stats["belief_available"] += 1
        if category:
            self.category_stats[category]["seen"] += 1

        if category not in set(self.args.categories):
            return model_action
        if int(obs_dict["action_mask"].sum()) <= 1:
            return model_action

        candidates, candidate_reasons = choose_hybrid_candidates(
            env,
            obs_dict,
            model_action,
            heuristic_action,
            belief_probs,
            self.args,
        )
        if len(candidates) <= 1:
            return model_action

        rows = evaluate_candidate_actions_belief(
            env,
            candidates,
            self.checkpoint,
            self.opponents,
            belief_probs,
            self.args,
            decision_seed,
        )
        by_action = {row["action"]: row for row in rows}
        model_row = by_action.get(model_action)
        if model_row is None:
            return model_action

        best = rows[0]
        margin = float(best["winrate"] - model_row["winrate"])
        reward_margin = float(best["mean_reward"] - model_row["mean_reward"])

        self.stats["search_checks"] += 1
        self.stats["sum_candidates"] += len(candidates)
        self.category_stats[category]["checked"] += 1
        self.category_stats[category]["sum_margin"] += margin

        if margin >= self.args.override_margin and reward_margin >= self.args.min_reward_margin:
            chosen = int(best["action"])
            if chosen != model_action:
                self.stats["overrides"] += 1
                self.stats["sum_override_margin"] += margin
                self.category_stats[category]["overrides"] += 1
                self.category_stats[category]["sum_override_margin"] += margin
                if len(self.examples) < self.args.example_limit:
                    self.examples.append(
                        {
                            "category": category,
                            "model_action": model_action,
                            "model_decoded": decode_action(model_action),
                            "heuristic_action": heuristic_action,
                            "heuristic_decoded": decode_action(heuristic_action),
                            "chosen_action": chosen,
                            "chosen_decoded": decode_action(chosen),
                            "margin": margin,
                            "reward_margin": reward_margin,
                            "belief_top": belief_top_summary(env, "player_0", belief_probs),
                            "candidate_reasons": candidate_reasons,
                            "top_actions": rows[:5],
                        }
                    )
            return chosen
        return model_action


def evaluate_player0_hybrid(checkpoint, opponents, games, seed_start, args):
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
        player = HybridSearchPlayer0(checkpoint, opponents, args)
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
                action = player.act(env, obs_dict, decision_seed=seed * 100 + turn)
                actions0 += 1
            else:
                action = opponent_step(env, agent, obs_dict, opponents, bot)
            env.step(action)

        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
        aggregate_stats.update(player.stats)
        for category, stats in player.category_stats.items():
            aggregate_categories[category].update(stats)
        if len(examples) < args.example_limit:
            examples.extend(player.examples[: args.example_limit - len(examples)])

    summary = summarize_rewards(rewards, wins, lengths)
    summary["hybrid_search"] = summarize_search(aggregate_stats, aggregate_categories)
    summary["examples"] = examples
    return summary


def make_uniform_args(args):
    return argparse.Namespace(
        categories=args.categories,
        rollouts_per_action=args.rollouts_per_action,
        max_actions=args.max_actions,
        override_margin=args.override_margin,
        min_reward_margin=args.min_reward_margin,
        player0_continuation=args.player0_continuation,
        example_limit=args.example_limit,
    )


def run_evaluation(checkpoint, games, seed_start, args, logger=None):
    started = perf_counter()
    hybrid_configs = {}
    step2_configs = {}
    uniform_configs = {}

    for name, opponents in OPPONENT_CONFIGS.items():
        cfg_start = perf_counter()
        hybrid_configs[name] = evaluate_player0_hybrid(checkpoint, opponents, games, seed_start, args)
        elapsed = perf_counter() - cfg_start
        if logger:
            logger.write(
                f"Config Step4 {name}",
                expected="Le champion hybride doit battre Step2 et idealement Step3 uniform-rollout.",
                actual=(
                    f"hybrid={hybrid_configs[name]['winrate']:.4f}, "
                    f"overrides={hybrid_configs[name]['hybrid_search']['overrides']}, "
                    f"elapsed={elapsed:.1f}s"
                ),
                details=hybrid_configs[name],
            )

        if args.compare_step2:
            step2_configs[name] = evaluate_player0_model(checkpoint, opponents, games, seed_start)

        if args.compare_uniform_step3:
            uniform_configs[name] = evaluate_player0_guided(
                checkpoint,
                opponents,
                games,
                seed_start,
                make_uniform_args(args),
            )

    hybrid_score = composite_score(hybrid_configs)
    step2_score = composite_score(step2_configs) if step2_configs else None
    uniform_score = composite_score(uniform_configs) if uniform_configs else None
    report = {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "games": games,
        "seed_start": seed_start,
        "elapsed_seconds": perf_counter() - started,
        "args": vars(args),
        "hybrid_configs": hybrid_configs,
        "hybrid_composite": hybrid_score,
        "step2_configs": step2_configs or None,
        "step2_composite": step2_score,
        "hybrid_minus_step2": (hybrid_score - step2_score) if step2_score is not None else None,
        "uniform_step3_configs": uniform_configs or None,
        "uniform_step3_composite": uniform_score,
        "hybrid_minus_uniform_step3": (hybrid_score - uniform_score) if uniform_score is not None else None,
    }
    return report


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate Step4 actor+belief+rollout hybrid champion.")
    parser.add_argument("--checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=800000)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["guard", "priest", "spy", "king", "prince", "chancellor_card", "chancellor_choice", "baron"],
    )
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--belief-candidate-pool", type=int, default=18)
    parser.add_argument("--override-margin", type=float, default=0.12)
    parser.add_argument("--min-reward-margin", type=float, default=-999.0)
    parser.add_argument("--belief-power", type=float, default=1.25)
    parser.add_argument("--belief-floor", type=float, default=0.02)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--compare-step2", action="store_true")
    parser.add_argument("--compare-uniform-step3", action="store_true")
    parser.add_argument("--example-limit", type=int, default=25)
    parser.add_argument("--output", default="step4_hybrid_search_eval.json")
    parser.add_argument("--run-log", default="step4_hybrid_champion/logs/2026-04-25_step4_hybrid_search.md")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut evaluation Step4 hybride",
        expected=(
            "Construire un joueur actor+belief+rollouts: Step2 decide par defaut, "
            "le belief guide les candidats et les determinizations, les rollouts tranchent."
        ),
        actual=f"checkpoint={checkpoint}, games={args.games}",
        details=vars(args),
    )

    report = run_evaluation(checkpoint, args.games, args.seed_start, args, logger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin evaluation Step4 hybride",
        expected="Verdict: Step4 doit ameliorer Step2 et clarifier son delta vs Step3.",
        actual=(
            f"hybrid_score={report['hybrid_composite']:.5f}, "
            f"delta_step2={report['hybrid_minus_step2']}, "
            f"delta_uniform={report['hybrid_minus_uniform_step3']}, "
            f"elapsed={report['elapsed_seconds']:.1f}s"
        ),
        details={
            "hybrid": arena_summary(report["hybrid_configs"]),
            "step2": arena_summary(report["step2_configs"]) if report["step2_configs"] else None,
            "uniform_step3": arena_summary(report["uniform_step3_configs"])
            if report["uniform_step3_configs"]
            else None,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

