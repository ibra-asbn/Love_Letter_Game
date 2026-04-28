"""Benchmark the step2 model against curriculum_phase1 and tactical KPIs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import absolute_to_relative_mask, relative_to_absolute_action
from step2_rl_finetune.common import (
    ExperimentLogger,
    STEP_REPORT_DIR,
    composite_score,
    ensure_step_dirs,
    now_stamp,
    resolve_checkpoint,
    resolve_step_path,
)


CARD_NAMES = {
    0: "Espionne",
    1: "Garde",
    2: "Pretre",
    3: "Baron",
    4: "Servante",
    5: "Prince",
    6: "Chancelier",
    7: "Roi",
    8: "Comtesse",
    9: "Princesse",
}

OPPONENT_CONFIGS = {
    "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
    "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
    "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
    "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
}


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def decode_action(action):
    if action >= 900:
        return {"kind": "chancellor_choice", "action": int(action)}
    return {
        "kind": "card",
        "card": int(action // 100),
        "target_idx": int((action % 100) // 10),
        "guess": int(action % 10),
    }


def non_princess_value(card):
    return -0.5 if card == 9 else card


def chancellor_kept(pool, action):
    if not pool:
        return None, None
    action_idx = action - 900
    if len(pool) >= 3:
        keep_idx = action_idx // 2
    elif len(pool) == 2:
        keep_idx = action_idx
    else:
        keep_idx = 0
    if keep_idx < 0 or keep_idx >= len(pool):
        keep_idx = 0
    return keep_idx, pool[keep_idx]


def extract_belief(policy):
    debug = getattr(policy, "last_debug", None)
    if debug is None:
        return None
    if hasattr(debug, "belief_probs") and debug.belief_probs is not None:
        return debug.belief_probs.squeeze(0).numpy()
    if hasattr(debug, "probs") and debug.probs is not None:
        return debug.probs
    return None


def target_rel_dim(my_idx, target_idx):
    if target_idx >= 4 or target_idx == my_idx:
        return None
    rel = (target_idx - my_idx) % 4
    return rel - 1 if rel else None


def best_guard_guess_from_belief(belief, my_idx, target_idx):
    if belief is None:
        return None
    dim = target_rel_dim(my_idx, target_idx)
    if dim is None:
        return None
    probs = np.array(belief[dim], copy=True)
    probs[1] = -1.0
    return int(np.argmax(probs))


def summarize_binary(total, count):
    return {
        "count": int(count),
        "total": int(total),
        "rate": float(count / total) if total else None,
    }


class TacticalStats:
    def __init__(self):
        self.card_counts = Counter()
        self.guard_attempts = 0
        self.guard_hits = 0
        self.guard_top_belief = 0
        self.guard_guess_counts = Counter()
        self.baron_attempts = 0
        self.baron_favorable = 0
        self.baron_neutral = 0
        self.baron_bad = 0
        self.prince_attempts = 0
        self.prince_princess_hits = 0
        self.prince_opp_high_hits = 0
        self.prince_self_low = 0
        self.prince_self_bad = 0
        self.king_attempts = 0
        self.king_better = 0
        self.king_equal = 0
        self.king_worse = 0
        self.chancellor_choices = 0
        self.chancellor_best_keep = 0
        self.countess_voluntary = 0
        self.examples = defaultdict(list)

    def record(self, env, agent, obs_dict, action, hand_before, belief):
        my_idx = int(agent.rsplit("_", 1)[1])
        decoded = decode_action(action)
        if decoded["kind"] == "chancellor_choice":
            pool = list(env._chancellor_pool)
            _keep_idx, kept = chancellor_kept(pool, action)
            if kept is not None:
                self.card_counts["ChancellorChoice"] += 1
                self.chancellor_choices += 1
                best = max(pool, key=non_princess_value) if pool else None
                if kept == best:
                    self.chancellor_best_keep += 1
                elif len(self.examples["chancellor_not_best"]) < 5:
                    self.examples["chancellor_not_best"].append(
                        {
                            "pool": [CARD_NAMES[c] for c in pool],
                            "kept": CARD_NAMES[kept],
                            "best": CARD_NAMES[best] if best is not None else None,
                        }
                    )
            return

        card = decoded["card"]
        target_idx = decoded["target_idx"]
        target = f"player_{target_idx}" if target_idx < 4 else None
        target_hand = list(env._hands.get(target, [])) if target else []
        target_card = target_hand[0] if target_hand else None
        remaining = list(hand_before)
        if card in remaining:
            remaining.remove(card)
        kept = remaining[0] if remaining else None
        self.card_counts[CARD_NAMES.get(card, str(card))] += 1

        if card == 1 and target_card is not None:
            guess = decoded["guess"]
            self.guard_attempts += 1
            self.guard_guess_counts[CARD_NAMES.get(guess, str(guess))] += 1
            if guess == target_card:
                self.guard_hits += 1
            top = best_guard_guess_from_belief(belief, my_idx, target_idx)
            if top is not None and guess == top:
                self.guard_top_belief += 1
            elif top is not None and len(self.examples["guard_not_top_belief"]) < 5:
                self.examples["guard_not_top_belief"].append(
                    {
                        "guess": CARD_NAMES.get(guess, str(guess)),
                        "target_card": CARD_NAMES.get(target_card, str(target_card)),
                        "top_belief": CARD_NAMES.get(top, str(top)),
                    }
                )

        elif card == 3 and target_card is not None and kept is not None:
            self.baron_attempts += 1
            if kept > target_card:
                self.baron_favorable += 1
            elif kept == target_card:
                self.baron_neutral += 1
            else:
                self.baron_bad += 1
                if len(self.examples["baron_bad"]) < 5:
                    self.examples["baron_bad"].append(
                        {
                            "kept": CARD_NAMES[kept],
                            "target_card": CARD_NAMES[target_card],
                            "hand_before": [CARD_NAMES[c] for c in hand_before],
                        }
                    )

        elif card == 5 and target_card is not None:
            self.prince_attempts += 1
            if target_idx == my_idx:
                if target_card <= 2:
                    self.prince_self_low += 1
                if target_card >= 7:
                    self.prince_self_bad += 1
                    if len(self.examples["prince_self_bad"]) < 5:
                        self.examples["prince_self_bad"].append({"discarded": CARD_NAMES[target_card]})
            else:
                if target_card == 9:
                    self.prince_princess_hits += 1
                if target_card >= 7:
                    self.prince_opp_high_hits += 1

        elif card == 7 and target_card is not None and kept is not None:
            self.king_attempts += 1
            if target_card > kept:
                self.king_better += 1
            elif target_card == kept:
                self.king_equal += 1
            else:
                self.king_worse += 1
                if len(self.examples["king_worse"]) < 5:
                    self.examples["king_worse"].append(
                        {"gave": CARD_NAMES[kept], "received": CARD_NAMES[target_card]}
                    )

        elif card == 8 and not (5 in hand_before or 7 in hand_before):
            self.countess_voluntary += 1

    def to_dict(self):
        return {
            "card_counts": dict(self.card_counts),
            "guard": {
                "attempts": int(self.guard_attempts),
                "hits": int(self.guard_hits),
                "hit_rate": float(self.guard_hits / self.guard_attempts) if self.guard_attempts else None,
                "top_belief_count": int(self.guard_top_belief),
                "top_belief_rate": float(self.guard_top_belief / self.guard_attempts)
                if self.guard_attempts
                else None,
                "guess_counts": dict(self.guard_guess_counts),
            },
            "baron": {
                "attempts": int(self.baron_attempts),
                "favorable": summarize_binary(self.baron_attempts, self.baron_favorable),
                "neutral": summarize_binary(self.baron_attempts, self.baron_neutral),
                "bad": summarize_binary(self.baron_attempts, self.baron_bad),
            },
            "prince": {
                "attempts": int(self.prince_attempts),
                "princess_hits": summarize_binary(self.prince_attempts, self.prince_princess_hits),
                "opponent_high_hits": summarize_binary(self.prince_attempts, self.prince_opp_high_hits),
                "self_low_redraws": summarize_binary(self.prince_attempts, self.prince_self_low),
                "self_bad_discards": summarize_binary(self.prince_attempts, self.prince_self_bad),
            },
            "king": {
                "attempts": int(self.king_attempts),
                "better_exchange": summarize_binary(self.king_attempts, self.king_better),
                "equal_exchange": summarize_binary(self.king_attempts, self.king_equal),
                "worse_exchange": summarize_binary(self.king_attempts, self.king_worse),
            },
            "chancellor": {
                "choices": int(self.chancellor_choices),
                "best_keep": summarize_binary(self.chancellor_choices, self.chancellor_best_keep),
            },
            "countess": {
                "voluntary_without_force": int(self.countess_voluntary),
            },
            "examples": {name: values for name, values in self.examples.items()},
        }


class RelativeModelSeat:
    """Use every learned policy in a seat-relative action convention."""

    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None
        self.last_belief = None

    def act(self, obs_dict, agent):
        my_idx = int(agent.rsplit("_", 1)[1])
        relative_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        # agent_id intentionally stays player_0: the action ids passed to the
        # model are relative, so old belief-aware checkpoints should interpret
        # target 1/2/3 as relative opponents, not absolute seats.
        relative_action, self.state = self.policy.act(relative_obs, self.state, agent_id="player_0")
        self.last_belief = extract_belief(self.policy)
        return relative_to_absolute_action(relative_action, my_idx)


def summarize_rewards(rewards, wins, lengths=None):
    rewards = np.asarray(rewards, dtype=np.float32)
    wins = np.asarray(wins, dtype=np.int32)
    winrate = float(wins.mean()) if len(wins) else 0.0
    out = {
        "games": int(len(wins)),
        "wins": int(wins.sum()),
        "winrate": winrate,
        "winrate_ci95": float(1.96 * np.sqrt(winrate * (1.0 - winrate) / max(1, len(wins)))),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "reward_ci95": float(1.96 * rewards.std() / np.sqrt(max(1, len(rewards)))) if len(rewards) else 0.0,
    }
    if lengths is not None:
        out["avg_actions_player0"] = float(np.mean(lengths)) if lengths else 0.0
    return out


def evaluate_player0_model(checkpoint, opponents, games, seed_start, collect_tactics):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    rewards = []
    wins = []
    lengths = []
    tactics = TacticalStats()

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        model = RelativeModelSeat(checkpoint)
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
                hand_before = list(env._hands.get(agent, []))
                action = model.act(obs_dict, agent)
                if collect_tactics:
                    tactics.record(env, agent, obs_dict, action, hand_before, model.last_belief)
                actions0 += 1
            else:
                opponent = opponents[agent]
                if opponent == "heuristic":
                    action = bot.choose_action(env, agent)
                elif opponent == "random":
                    action = random_action(obs_dict)
                else:
                    raise ValueError(opponent)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
    out = summarize_rewards(rewards, wins, lengths)
    if collect_tactics:
        out["tactics"] = tactics.to_dict()
    return out


def evaluate_same_opponent_arena(models, games, seed_start, logger):
    arena = {}
    for model_name, checkpoint in models.items():
        arena[model_name] = {"configs": {}}
        for config_name, opponents in OPPONENT_CONFIGS.items():
            logger.write(
                f"Arena {model_name} - {config_name}",
                expected="Comparer les deux checkpoints en player_0 sur les memes seeds.",
            )
            arena[model_name]["configs"][config_name] = evaluate_player0_model(
                checkpoint,
                opponents,
                games,
                seed_start,
                collect_tactics=True,
            )
        arena[model_name]["composite"] = composite_score(arena[model_name]["configs"])
    return arena


def evaluate_role_table(models, roles, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    role_rewards = defaultdict(list)
    role_wins = defaultdict(list)

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        seats = {
            agent: RelativeModelSeat(models[role])
            for agent, role in roles.items()
            if role in models
        }
        rewards = {agent: 0.0 for agent in env.possible_agents}
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            role = roles[agent]
            if role == "random":
                action = random_action(obs_dict)
            elif role in models:
                action = seats[agent].act(obs_dict, agent)
            else:
                raise ValueError(role)
            env.step(action)
        for agent, role in roles.items():
            role_rewards[role].append(rewards[agent])
            role_wins[role].append(int(rewards[agent] >= 1.0))

    return {
        role: summarize_rewards(role_rewards[role], role_wins[role])
        for role in sorted(role_rewards)
    }


def run_direct_tables(models, games, seed_start, logger):
    tables = {
        "candidate_vs_1_curriculum_2R": {
            "player_0": "candidate",
            "player_1": "curriculum",
            "player_2": "random",
            "player_3": "random",
        },
        "candidate_vs_3_curriculum": {
            "player_0": "candidate",
            "player_1": "curriculum",
            "player_2": "curriculum",
            "player_3": "curriculum",
        },
        "curriculum_vs_3_candidate": {
            "player_0": "curriculum",
            "player_1": "candidate",
            "player_2": "candidate",
            "player_3": "candidate",
        },
        "two_candidate_two_curriculum": {
            "player_0": "candidate",
            "player_1": "curriculum",
            "player_2": "candidate",
            "player_3": "curriculum",
        },
    }
    results = {}
    for name, roles in tables.items():
        logger.write(
            f"Direct table {name}",
            expected="Mesurer les winrates quand les deux checkpoints coexistent a table.",
            details=roles,
        )
        results[name] = evaluate_role_table(models, roles, games, seed_start)
    return results


def compact_tactics(configs):
    merged = {
        "guard_attempts": 0,
        "guard_hits": 0,
        "guard_top_belief": 0,
        "baron_attempts": 0,
        "baron_favorable": 0,
        "baron_bad": 0,
        "prince_attempts": 0,
        "prince_princess_hits": 0,
        "king_attempts": 0,
        "king_better": 0,
        "king_worse": 0,
        "chancellor_choices": 0,
        "chancellor_best_keep": 0,
    }
    for values in configs.values():
        tactics = values.get("tactics", {})
        guard = tactics.get("guard", {})
        baron = tactics.get("baron", {})
        prince = tactics.get("prince", {})
        king = tactics.get("king", {})
        chancellor = tactics.get("chancellor", {})
        merged["guard_attempts"] += guard.get("attempts", 0)
        merged["guard_hits"] += guard.get("hits", 0)
        merged["guard_top_belief"] += guard.get("top_belief_count", 0)
        merged["baron_attempts"] += baron.get("attempts", 0)
        merged["baron_favorable"] += baron.get("favorable", {}).get("count", 0)
        merged["baron_bad"] += baron.get("bad", {}).get("count", 0)
        merged["prince_attempts"] += prince.get("attempts", 0)
        merged["prince_princess_hits"] += prince.get("princess_hits", {}).get("count", 0)
        merged["king_attempts"] += king.get("attempts", 0)
        merged["king_better"] += king.get("better_exchange", {}).get("count", 0)
        merged["king_worse"] += king.get("worse_exchange", {}).get("count", 0)
        merged["chancellor_choices"] += chancellor.get("choices", 0)
        merged["chancellor_best_keep"] += chancellor.get("best_keep", {}).get("count", 0)
    return {
        "guard_hit_rate": float(merged["guard_hits"] / merged["guard_attempts"])
        if merged["guard_attempts"]
        else None,
        "guard_top_belief_rate": float(merged["guard_top_belief"] / merged["guard_attempts"])
        if merged["guard_attempts"]
        else None,
        "baron_favorable_rate": float(merged["baron_favorable"] / merged["baron_attempts"])
        if merged["baron_attempts"]
        else None,
        "baron_bad_rate": float(merged["baron_bad"] / merged["baron_attempts"])
        if merged["baron_attempts"]
        else None,
        "prince_princess_hit_rate": float(merged["prince_princess_hits"] / merged["prince_attempts"])
        if merged["prince_attempts"]
        else None,
        "king_better_rate": float(merged["king_better"] / merged["king_attempts"])
        if merged["king_attempts"]
        else None,
        "king_worse_rate": float(merged["king_worse"] / merged["king_attempts"])
        if merged["king_attempts"]
        else None,
        "chancellor_best_keep_rate": float(merged["chancellor_best_keep"] / merged["chancellor_choices"])
        if merged["chancellor_choices"]
        else None,
        "counts": merged,
    }


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Benchmark step2 against curriculum_phase1.")
    parser.add_argument("--candidate", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--curriculum", default="curriculum_phase1.pth")
    parser.add_argument("--arena-games", type=int, default=3000)
    parser.add_argument("--direct-games", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=760000)
    parser.add_argument("--output", default="step2_vs_curriculum_benchmark.json")
    parser.add_argument("--run-log", default="step2_rl_finetune/logs/2026-04-24_step2_vs_curriculum_benchmark.md")
    args = parser.parse_args()

    candidate = resolve_checkpoint(args.candidate)
    curriculum = resolve_checkpoint(args.curriculum)
    output = resolve_step_path(args.output, STEP_REPORT_DIR)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut benchmark curriculum",
        expected=(
            "Comparer le modele step2 a curriculum_phase1 sans en faire le centre de la pipeline: "
            "arena commune, tables directes, et KPIs tactiques Love Letter."
        ),
        actual=f"candidate={candidate}, curriculum={curriculum}",
        details=vars(args),
    )

    models = {"candidate": candidate, "curriculum": curriculum}
    arena = evaluate_same_opponent_arena(models, args.arena_games, args.seed_start, logger)
    direct = run_direct_tables(models, args.direct_games, args.seed_start + 100_000, logger)
    tactical_summary = {
        name: compact_tactics(values["configs"])
        for name, values in arena.items()
    }

    report = {
        "created_at": now_stamp(),
        "candidate": str(candidate),
        "curriculum": str(curriculum),
        "arena_games": args.arena_games,
        "direct_games": args.direct_games,
        "seed_start": args.seed_start,
        "arena": arena,
        "direct_tables": direct,
        "tactical_summary": tactical_summary,
        "notes": [
            "Arena commune: comparaison la plus propre, chaque checkpoint joue player_0 contre les memes compositions.",
            "Tables directes: indicatives, car un jeu a 4 joueurs est sensible au siege et aux adversaires presents.",
            "Les learned policies sont appelees avec une convention d'action relative pour eviter un biais de siege.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin benchmark curriculum",
        expected="Obtenir une lecture claire: step2 bat-il curriculum, et ou tactiquement ?",
        actual=(
            f"candidate_composite={arena['candidate']['composite']:.5f}, "
            f"curriculum_composite={arena['curriculum']['composite']:.5f}"
        ),
        details={
            "tactical_summary": tactical_summary,
            "direct_tables": direct,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

