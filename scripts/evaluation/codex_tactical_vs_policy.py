"""Run a small Codex tactical-policy match against three policy clones."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_policy import load_belief_policy
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path


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

MAX_COUNTS = np.array([2, 6, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.float32)
GUARD_GUESSES = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def card_label(card):
    return f"{CARD_NAMES[card]} ({card})"


def decode_action(action):
    if action >= 900:
        return f"{action}: choisit sa carte (Chancelier)"
    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    if card == 1 and target_idx < 4:
        return f"{action}: joue {card_label(card)} sur player_{target_idx}, devine {card_label(guess)}"
    if card in (2, 3, 5, 7) and target_idx < 4:
        return f"{action}: joue {card_label(card)} sur player_{target_idx}"
    return f"{action}: joue {card_label(card)}"


def valid_actions(obs_dict):
    return [int(a) for a in np.where(obs_dict["action_mask"] == 1)[0]]


def known_card(env, agent, target):
    known = env._known_cards[agent][target]
    idx = np.where(known >= 1.0)[0]
    return int(idx[0]) if len(idx) else None


def remaining_distribution(env, agent):
    remaining = MAX_COUNTS.copy()
    for player in env.possible_agents:
        for card in env._played_cards.get(player, []):
            remaining[card] -= 1.0
    for card in env._hands.get(agent, []):
        remaining[card] -= 1.0
    for target in env.possible_agents:
        if target == agent:
            continue
        known = known_card(env, agent, target)
        if known is not None:
            remaining[known] -= 1.0
    remaining = np.clip(remaining, 0.0, None)
    if remaining.sum() <= 0:
        return np.ones(10, dtype=np.float32) / 10.0
    return remaining / remaining.sum()


def valid_targets(env, agent):
    return [
        idx
        for idx, target in enumerate(env.possible_agents)
        if target != agent
        and target in env.agents
        and not env.terminations.get(target, True)
        and not env._protected.get(target, False)
    ]


def chancellor_score(env, action):
    pool = list(env._chancellor_pool)
    if not pool:
        return 0.0
    action_idx = action - 900
    if len(pool) >= 3:
        keep_idx = action_idx // 2
    elif len(pool) == 2:
        keep_idx = action_idx
    else:
        keep_idx = 0
    if keep_idx >= len(pool):
        keep_idx = 0
    kept = pool[keep_idx]
    value = -0.5 if kept == 9 else kept
    return 100.0 + value


def score_action(env, agent, action):
    if action >= 900:
        return chancellor_score(env, action)

    hand = list(env._hands.get(agent, []))
    deck_left = len(env._deck)
    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    target = f"player_{target_idx}" if target_idx < 4 else None

    kept = list(hand)
    if card in kept:
        kept.remove(card)
    kept_card = kept[0] if kept else None

    probs = remaining_distribution(env, agent)
    score = 0.0

    if card == 0:
        score = 40.0 if env._played_cards.get(agent, []).count(0) == 0 else 16.0

    elif card == 1:
        if target is None:
            return -100.0
        known = known_card(env, agent, target)
        if known is not None and known in GUARD_GUESSES:
            score = 140.0 if guess == known else -40.0
        else:
            guess_probs = probs.copy()
            guess_probs[1] = 0.0
            best = int(np.argmax(guess_probs))
            target_bonus = 2.0 * target_idx
            score = 12.0 + 120.0 * float(guess_probs[guess]) + target_bonus
            if guess == best:
                score += 22.0

    elif card == 2:
        if target is None:
            return -100.0
        known = known_card(env, agent, target)
        score = 48.0 if known is None else 18.0
        score += 2.0 * target_idx

    elif card == 3:
        if kept_card is None or target is None:
            return -100.0
        known = known_card(env, agent, target)
        if known is not None:
            if kept_card > known:
                score = 110.0
            elif kept_card < known:
                score = -80.0
            else:
                score = -5.0
        elif kept_card <= 4:
            score = -65.0
        else:
            p_lower = probs[:kept_card].sum()
            p_higher = probs[kept_card + 1 :].sum() if kept_card < 9 else 0.0
            score = 35.0 + 85.0 * float(p_lower) - 120.0 * float(p_higher)

    elif card == 4:
        score = 34.0 if deck_left <= 6 else 22.0

    elif card == 5:
        if target is None:
            return -100.0
        known = known_card(env, agent, target)
        my_idx = env.possible_agents.index(agent)
        if target_idx == my_idx:
            if kept_card == 9:
                score = -100.0
            elif kept_card is not None and kept_card <= 2:
                score = 60.0
            elif kept_card is not None and kept_card <= 4:
                score = 35.0
            else:
                score = 5.0
        elif known == 9:
            score = 140.0
        else:
            score = 30.0 if deck_left <= 5 else 8.0
            if known is not None and known >= 7:
                score += 30.0

    elif card == 6:
        other = kept_card
        if other is None:
            score = 45.0
        elif other <= 3:
            score = 58.0
        elif other <= 5:
            score = 36.0
        else:
            score = 12.0

    elif card == 7:
        if target is None or kept_card is None:
            return -100.0
        known = known_card(env, agent, target)
        if known is not None and known > kept_card:
            score = 90.0
        elif deck_left <= 5 and kept_card <= 4:
            score = 38.0
        else:
            score = -30.0

    elif card == 8:
        score = 18.0 if (5 in hand or 7 in hand) else -20.0

    elif card == 9:
        score = -200.0

    return float(score)


def choose_codex_action(env, agent, obs_dict):
    scored = sorted(
        [(score_action(env, agent, action), action) for action in valid_actions(obs_dict)],
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    return scored[0][1], scored[:5]


def run(checkpoint, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    opponents = {f"player_{idx}": load_belief_policy(checkpoint) for idx in range(1, 4)}
    rewards = []
    lines = [
        "Player_0: Codex manual tactical policy, chosen for this run",
        "Information constraint: no opponent hidden hands used",
        f"Opponents: 3 copies of {checkpoint}",
        f"Games: {games}",
        "",
    ]

    for game_idx in range(games):
        seed = seed_start + game_idx
        env.reset(seed=seed)
        states = {agent: None for agent in opponents}
        reward_total = 0.0
        turn_logs = []

        for _agent in env.agent_iter():
            obs_dict, reward, term, trunc, _info = env.last()
            if _agent == "player_0":
                reward_total += float(reward)
            if term or trunc:
                env.step(None)
                continue

            if _agent == "player_0":
                hand = [card_label(card) for card in env._hands.get(_agent, [])]
                action, top = choose_codex_action(env, _agent, obs_dict)
                turn_logs.append(
                    {
                        "hand": hand,
                        "deck": len(env._deck),
                        "action": action,
                        "top": top,
                    }
                )
            else:
                action, states[_agent] = opponents[_agent].act(obs_dict, states[_agent], agent_id=_agent)

            env.step(action)

        rewards.append(reward_total)
        won = reward_total >= 1.0
        lines.append(f"Game {game_idx + 1} seed={seed} reward={reward_total:+.1f} won={won}")
        lines.append(
            f"  winners={getattr(env, '_round_winners', [])} "
            f"reason={getattr(env, '_round_win_reason', None)} spy={getattr(env, '_round_spy_winner', None)}"
        )
        for turn_idx, turn in enumerate(turn_logs, start=1):
            lines.append(f"  Turn {turn_idx}: hand={turn['hand']} deck={turn['deck']}")
            lines.append(f"    chose {decode_action(turn['action'])}")
            lines.append("    top considered:")
            for score, action in turn["top"]:
                lines.append(f"      score={score:.1f} action={decode_action(action)}")
        lines.append("")

    wins = sum(1 for reward in rewards if reward >= 1.0)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    lines.append(f"Summary: wins={wins}/{games} winrate={wins / games:.1%} mean_reward={mean_reward:.3f}")
    return "\n".join(lines), wins / games, mean_reward


def main():
    parser = argparse.ArgumentParser(description="Codex tactical policy vs three policy clones.")
    parser.add_argument("--checkpoint", default=str(checkpoint_path("curriculum_phase1.pth")))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=70000)
    parser.add_argument("--output-log", required=True)
    args = parser.parse_args()

    log, winrate, mean_reward = run(args.checkpoint, args.games, args.seed_start)
    output = Path(args.output_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(log, encoding="utf-8")
    print(log)
    print(f"Wrote {output}")
    print(f"winrate={winrate:.3f} mean_reward={mean_reward:.3f}")


if __name__ == "__main__":
    main()
