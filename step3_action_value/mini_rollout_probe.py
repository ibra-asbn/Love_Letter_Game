"""Mini action-value probe for critical Love Letter decisions.

This is intentionally a probe, not the final training pipeline. It samples a
small set of critical states, evaluates legal actions through determinized
rollouts, and reports whether rollout labels look useful enough for step 3.
"""

from __future__ import annotations

import argparse
import copy
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
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint


STEP_DIR = PROJECT_ROOT / "step3_action_value"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"

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

CRITICAL_ORDER = ["baron", "king", "chancellor_card", "prince", "guard"]


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def decode_action(action):
    if 900 <= action <= 905:
        return {
            "kind": "chancellor_choice",
            "action": int(action),
            "label": f"ChancellorChoice:{int(action)}",
        }
    card = int(action // 100)
    target = int((action % 100) // 10)
    guess = int(action % 10)
    if card == 1:
        label = f"Garde->p{target}:{CARD_NAMES.get(guess, guess)}"
    elif card in {2, 3, 5, 7}:
        label = f"{CARD_NAMES.get(card, card)}->p{target}"
    else:
        label = CARD_NAMES.get(card, str(card))
    return {
        "kind": "card",
        "action": int(action),
        "card": card,
        "card_name": CARD_NAMES.get(card, str(card)),
        "target": target,
        "guess": guess,
        "guess_name": CARD_NAMES.get(guess, str(guess)),
        "label": label,
    }


class RelativeModelSeat:
    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None

    def act(self, obs_dict, agent):
        my_idx = int(agent.rsplit("_", 1)[1])
        relative_obs = {
            "observation": obs_dict["observation"],
            "action_mask": absolute_to_relative_mask(obs_dict["action_mask"], my_idx),
        }
        relative_action, self.state = self.policy.act(relative_obs, self.state, agent_id="player_0")
        return relative_to_absolute_action(relative_action, my_idx)


def classify_state(env, agent):
    if env._chancellor_pending and env.agent_selection == agent:
        return "chancellor_choice"
    hand = list(env._hands.get(agent, []))
    if 3 in hand:
        return "baron"
    if 7 in hand:
        return "king"
    if 6 in hand:
        return "chancellor_card"
    if 5 in hand:
        return "prince"
    if 1 in hand:
        return "guard"
    if 2 in hand:
        return "priest"
    if 0 in hand:
        return "spy"
    if 4 in hand:
        return "handmaid"
    if 8 in hand:
        return "countess"
    if 9 in hand:
        return "princess"
    return None


def known_card_for(env, observer, opp):
    known = env._known_cards[observer][opp]
    idx = np.where(known >= 1.0)[0]
    return int(idx[0]) if len(idx) else None


def remove_card(counts, card):
    counts[int(card)] -= 1
    if counts[int(card)] < -1e-6:
        raise ValueError(f"Negative card count while determinizing: {card}")


def draw_from_counts(counts, rng):
    available = np.flatnonzero(counts > 0)
    if len(available) == 0:
        return None
    probs = counts[available].astype(np.float64)
    probs = probs / probs.sum()
    card = int(rng.choice(available, p=probs))
    remove_card(counts, card)
    return card


def determinize_for_player(base_env, observer="player_0", seed=0):
    """Sample hidden hands/deck consistent with public info and observer knowledge."""
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

    # Preserve known opponent cards from observer knowledge; sample the rest.
    for opp in env.possible_agents:
        if opp == observer or env.terminations.get(opp, False) or not env._hands.get(opp):
            continue
        known = known_card_for(env, observer, opp)
        if known is not None and counts[int(known)] > 0:
            env._hands[opp] = [known]
            remove_card(counts, known)
        else:
            sampled = draw_from_counts(counts, rng)
            env._hands[opp] = [sampled] if sampled is not None else []

    if env._set_aside is not None:
        env._set_aside = draw_from_counts(counts, rng)

    deck_len = len(env._deck)
    deck = [None] * deck_len
    observer_knowledge = env._deck_knowledge.get(observer, {})
    for pos, card in observer_knowledge.items():
        idx = deck_len - 1 - int(pos)
        if 0 <= idx < deck_len and deck[idx] is None and counts[int(card)] > 0:
            deck[idx] = int(card)
            remove_card(counts, card)

    for idx in range(deck_len):
        if deck[idx] is None:
            deck[idx] = draw_from_counts(counts, rng)
    env._deck = [int(card) for card in deck if card is not None]
    return env


def continuation_action(env, agent, obs_dict, bot, model_seats, continuation):
    if continuation == "heuristic":
        return bot.choose_action(env, agent)
    if continuation == "model":
        if agent not in model_seats:
            raise ValueError(f"Missing model seat for {agent}")
        return model_seats[agent].act(obs_dict, agent)
    if continuation == "random":
        return random_action(obs_dict)
    raise ValueError(continuation)


def rollout_once(base_env, first_action, seed, player0_continuation, opponent_policy, checkpoint):
    env = determinize_for_player(base_env, "player_0", seed)
    bot = HeuristicBot()
    reward0 = 0.0

    # Apply the candidate action from the sampled hidden state.
    obs_dict, reward, terminated, truncated, _info = env.last()
    if env.agent_selection == "player_0":
        reward0 += float(reward)
    if terminated or truncated:
        env.step(None)
    else:
        env.step(first_action)

    model_seats = {}
    if player0_continuation == "model":
        model_seats["player_0"] = RelativeModelSeat(checkpoint)
    if opponent_policy == "model":
        for idx in range(1, 4):
            model_seats[f"player_{idx}"] = RelativeModelSeat(checkpoint)

    for agent in env.agent_iter():
        obs_dict, reward, terminated, truncated, _info = env.last()
        if agent == "player_0":
            reward0 += float(reward)
        if terminated or truncated:
            env.step(None)
            continue
        if agent == "player_0":
            action = continuation_action(env, agent, obs_dict, bot, model_seats, player0_continuation)
        else:
            if opponent_policy == "heuristic":
                action = bot.choose_action(env, agent)
            elif opponent_policy == "random":
                action = random_action(obs_dict)
            elif opponent_policy == "model":
                action = model_seats[agent].act(obs_dict, agent)
            else:
                raise ValueError(opponent_policy)
        env.step(action)
    return reward0, int(reward0 >= 1.0)


def evaluate_actions(base_env, actions, args, checkpoint):
    rows = []
    for action in actions:
        rewards = []
        wins = []
        for rollout in range(args.rollouts_per_action):
            seed = args.seed * 1_000_000 + action * 1000 + rollout
            reward, win = rollout_once(
                base_env,
                int(action),
                seed,
                args.player0_continuation,
                args.opponent_policy,
                checkpoint,
            )
            rewards.append(reward)
            wins.append(win)
        winrate = float(np.mean(wins)) if wins else 0.0
        rows.append(
            {
                "action": int(action),
                "decoded": decode_action(int(action)),
                "winrate": winrate,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "reward_std": float(np.std(rewards)) if rewards else 0.0,
                "wins": int(sum(wins)),
                "rollouts": int(len(wins)),
            }
        )
    rows.sort(key=lambda row: (row["winrate"], row["mean_reward"]), reverse=True)
    return rows


def compact_state(env, agent, category, model_action, heuristic_action):
    obs = env.observe(agent)
    valid = np.where(obs["action_mask"] == 1)[0].tolist()
    known = {}
    for opp in env.possible_agents:
        if opp == agent:
            continue
        card = known_card_for(env, agent, opp)
        if card is not None:
            known[opp] = CARD_NAMES[card]
    return {
        "category": category,
        "agent": agent,
        "hand": [CARD_NAMES[c] for c in env._hands.get(agent, [])],
        "hand_ids": [int(c) for c in env._hands.get(agent, [])],
        "deck_size": len(env._deck),
        "protected": {a: bool(env._protected.get(a, False)) for a in env.possible_agents},
        "known_cards": known,
        "played_counts": {
            a: [CARD_NAMES[c] for c in env._played_cards.get(a, [])]
            for a in env.possible_agents
        },
        "valid_action_count": len(valid),
        "model_action": int(model_action),
        "model_decoded": decode_action(int(model_action)),
        "heuristic_action": int(heuristic_action),
        "heuristic_decoded": decode_action(int(heuristic_action)),
    }


def collect_states(args, checkpoint, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    collected = []
    counts = Counter()

    for game in range(args.collect_games):
        if all(counts[c] >= args.states_per_category for c in args.categories):
            break
        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        model = RelativeModelSeat(checkpoint)
        for _turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                category = classify_state(env, agent)
                model_action = model.act(obs_dict, agent)
                if (
                    category in args.categories
                    and counts[category] < args.states_per_category
                    and len(np.where(obs_dict["action_mask"] == 1)[0]) > 1
                ):
                    heuristic_action = bot.choose_action(env, agent)
                    collected.append(
                        {
                            "game_seed": seed,
                            "env": copy.deepcopy(env),
                            "summary": compact_state(env, agent, category, model_action, heuristic_action),
                        }
                    )
                    counts[category] += 1
                    logger.write(
                        f"Etat collecte - {category}",
                        expected="Capturer un etat critique pour tester des actions concurrentes.",
                        actual=(
                            f"seed={seed}, hand={collected[-1]['summary']['hand']}, "
                            f"model={collected[-1]['summary']['model_decoded']['label']}"
                        ),
                    )
                action = model_action
            else:
                action = bot.choose_action(env, agent) if args.collect_opponents == "heuristic" else random_action(obs_dict)
            env.step(action)
    return collected


def choose_actions_for_probe(env, max_actions):
    obs = env.observe("player_0")
    valid = [int(a) for a in np.where(obs["action_mask"] == 1)[0]]
    if len(valid) <= max_actions:
        return valid

    # Prefer broad card coverage if a state has too many Guard guesses.
    by_card = defaultdict(list)
    for action in valid:
        card = 99 if action >= 900 else action // 100
        by_card[card].append(action)
    selected = []
    for card in sorted(by_card):
        selected.extend(by_card[card][: max(1, max_actions // max(1, len(by_card)))])
    for action in valid:
        if len(selected) >= max_actions:
            break
        if action not in selected:
            selected.append(action)
    return selected[:max_actions]


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Mini action-value rollout probe.")
    parser.add_argument("--checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=500)
    parser.add_argument("--states-per-category", type=int, default=1)
    parser.add_argument("--categories", nargs="+", default=CRITICAL_ORDER)
    parser.add_argument("--rollouts-per-action", type=int, default=80)
    parser.add_argument("--max-actions", type=int, default=30)
    parser.add_argument("--collect-opponents", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--opponent-policy", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--player0-continuation", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--output", default="mini_rollout_probe.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-24_mini_rollout_probe.md")
    parser.add_argument("--seed", type=int, default=9100)
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = Path(args.output)
    if output.parent == Path("."):
        output = REPORT_DIR / output
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut mini action-value probe",
        expected=(
            "Verifier si des rollouts determinises donnent des preferences d'actions exploitables "
            "sur des etats critiques."
        ),
        actual=f"checkpoint={checkpoint}",
        details=vars(args),
    )

    states = collect_states(args, checkpoint, logger)
    results = []
    for idx, item in enumerate(states, start=1):
        env = item["env"]
        actions = choose_actions_for_probe(env, args.max_actions)
        rows = evaluate_actions(env, actions, args, checkpoint)
        model_action = item["summary"]["model_action"]
        heuristic_action = item["summary"]["heuristic_action"]
        by_action = {row["action"]: row for row in rows}
        best = rows[0] if rows else None
        model_row = by_action.get(model_action)
        heuristic_row = by_action.get(heuristic_action)
        result = {
            "index": idx,
            "state": item["summary"],
            "num_actions_evaluated": len(actions),
            "best_action": best,
            "model_action_value": model_row,
            "heuristic_action_value": heuristic_row,
            "model_regret_winrate": (best["winrate"] - model_row["winrate"]) if best and model_row else None,
            "heuristic_regret_winrate": (best["winrate"] - heuristic_row["winrate"]) if best and heuristic_row else None,
            "top_actions": rows[:8],
        }
        results.append(result)
        logger.write(
            f"Probe etat {idx} - {item['summary']['category']}",
            expected="Voir si l'action choisie est proche du top rollout ou si un autre coup domine.",
            actual=(
                f"best={best['decoded']['label'] if best else None}, "
                f"model={item['summary']['model_decoded']['label']}, "
                f"regret={result['model_regret_winrate']}"
            ),
            details={
                "state": item["summary"],
                "top_actions": result["top_actions"],
                "model_action_value": model_row,
                "heuristic_action_value": heuristic_row,
            },
        )

    regrets = [r["model_regret_winrate"] for r in results if r["model_regret_winrate"] is not None]
    report = {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "args": vars(args),
        "states_evaluated": len(results),
        "mean_model_regret_winrate": float(np.mean(regrets)) if regrets else None,
        "max_model_regret_winrate": float(np.max(regrets)) if regrets else None,
        "results": results,
        "limitations": [
            "Mini test seulement: peu d'etats et rollouts limites.",
            "Les determinisations approximent l'information de player_0, mais ne remplacent pas une vraie belief sampler.",
            "Le rollout utilise une politique de continuation configurable; ici ce n'est pas encore une Q-value parfaite.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin mini action-value probe",
        expected="Decider si cette approche merite une vraie etape 3.",
        actual=(
            f"states={len(results)}, mean_regret={report['mean_model_regret_winrate']}, "
            f"max_regret={report['max_model_regret_winrate']}"
        ),
        details={
            "summary": {
                "states_evaluated": len(results),
                "mean_model_regret_winrate": report["mean_model_regret_winrate"],
                "max_model_regret_winrate": report["max_model_regret_winrate"],
            },
            "best_vs_model": [
                {
                    "category": r["state"]["category"],
                    "hand": r["state"]["hand"],
                    "model": r["state"]["model_decoded"]["label"],
                    "best": r["best_action"]["decoded"]["label"] if r["best_action"] else None,
                    "regret": r["model_regret_winrate"],
                }
                for r in results
            ],
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
