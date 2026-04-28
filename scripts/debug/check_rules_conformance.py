"""Executable conformance checks for the local Love Letter rules.

These tests focus on rules that are easy to break when adding richer knowledge
state for learning agents.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.engine import LoveLetterRLEnv


def fresh_env() -> LoveLetterRLEnv:
    env = LoveLetterRLEnv(num_players=4)
    env.reset(seed=123, options={"starting_agent": "player_0"})
    env.agents = env.possible_agents[:]
    env.rewards = {agent: 0.0 for agent in env.possible_agents}
    env._cumulative_rewards = {agent: 0.0 for agent in env.possible_agents}
    env.terminations = {agent: False for agent in env.possible_agents}
    env.truncations = {agent: False for agent in env.possible_agents}
    env._deck = [0, 1, 2, 4, 6, 7, 8, 0]
    env._set_aside = 1
    env._hands = {
        "player_0": [1, 2],
        "player_1": [3],
        "player_2": [4],
        "player_3": [5],
    }
    env._played_cards = {agent: [] for agent in env.possible_agents}
    env._protected = {agent: False for agent in env.possible_agents}
    env._known_cards = {
        agent: {opp: np.zeros(10, dtype=np.float32) for opp in env.possible_agents}
        for agent in env.possible_agents
    }
    env._deck_knowledge = {agent: {} for agent in env.possible_agents}
    env._tokens = {agent: 0 for agent in env.possible_agents}
    env._turns_hand_unchanged = {agent: 0 for agent in env.possible_agents}
    env._min_kept_card = {agent: 0 for agent in env.possible_agents}
    env._countess_voluntary = {agent: False for agent in env.possible_agents}
    env._chancellor_pending = False
    env._chancellor_pool = []
    env._round_winners = []
    env._round_win_reason = None
    env._round_spy_winner = None
    env.agent_selection = "player_0"
    return env


def known_indices(env: LoveLetterRLEnv, observer: str, target: str) -> list[int]:
    return [idx for idx, value in enumerate(env._known_cards[observer][target]) if value >= 1.0]


def test_guard_wrong_does_not_reveal() -> None:
    env = fresh_env()
    env._hands["player_0"] = [1, 2]
    env._hands["player_1"] = [8]
    env.step(100 + 1 * 10 + 5)
    assert not env.terminations["player_1"]
    assert env._played_cards["player_1"] == []
    assert known_indices(env, "player_0", "player_1") == []


def test_priest_is_private_exact_info() -> None:
    env = fresh_env()
    env._hands["player_0"] = [2, 4]
    env._hands["player_1"] = [7]
    env.step(200 + 1 * 10)
    assert known_indices(env, "player_0", "player_1") == [7]
    assert known_indices(env, "player_2", "player_1") == []


def test_baron_win_public_lower_bound_not_exact_card() -> None:
    env = fresh_env()
    env._hands["player_0"] = [3, 6]
    env._hands["player_1"] = [2]
    env.step(300 + 1 * 10)
    assert env.terminations["player_1"]
    assert env._played_cards["player_1"] == [2]
    assert env._min_kept_card["player_0"] == 3
    assert known_indices(env, "player_2", "player_0") == []


def test_baron_loss_public_lower_bound_not_exact_card() -> None:
    env = fresh_env()
    env._hands["player_0"] = [3, 2]
    env._hands["player_1"] = [9]
    env.step(300 + 1 * 10)
    assert env.terminations["player_0"]
    assert env._played_cards["player_0"] == [3, 2]
    assert env._min_kept_card["player_1"] == 3
    assert known_indices(env, "player_2", "player_1") == []


def test_baron_tie_private_exact_info_for_both_players() -> None:
    env = fresh_env()
    env._hands["player_0"] = [3, 5]
    env._hands["player_1"] = [5]
    env.step(300 + 1 * 10)
    assert not env.terminations["player_0"]
    assert not env.terminations["player_1"]
    assert known_indices(env, "player_0", "player_1") == [5]
    assert known_indices(env, "player_1", "player_0") == [5]
    assert known_indices(env, "player_2", "player_0") == []
    assert env._min_kept_card["player_0"] == 0
    assert env._min_kept_card["player_1"] == 0


def test_handmaid_prevents_targeting_even_if_action_is_forced_in() -> None:
    env = fresh_env()
    env.agent_selection = "player_1"
    env._protected["player_0"] = True
    env._hands["player_0"] = [5]
    env._hands["player_1"] = [1, 2]
    env.step(100 + 0 * 10 + 5)
    assert not env.terminations["player_0"]
    assert env._hands["player_0"] == [5]


def test_prince_self_when_all_opponents_protected_can_discard_princess() -> None:
    env = fresh_env()
    env._hands["player_0"] = [5, 9]
    for opp in ["player_1", "player_2", "player_3"]:
        env._protected[opp] = True
    mask = env.observe("player_0")["action_mask"]
    assert mask[500] == 1
    env.step(500)
    assert env.terminations["player_0"]
    assert 9 in env._played_cards["player_0"]


def test_chancellor_can_bottom_princess_without_elimination() -> None:
    env = fresh_env()
    env._hands["player_0"] = [6, 9]
    env._deck = [4, 2]
    env.step(690)
    assert env._chancellor_pending
    assert env._chancellor_pool == [9, 2, 4]
    env.step(905)
    assert not env.terminations["player_0"]
    assert env._hands["player_0"] == [4]
    assert env._deck and env._deck[0] == 9


def test_king_is_private_bilateral_info() -> None:
    env = fresh_env()
    env._hands["player_0"] = [7, 2]
    env._hands["player_1"] = [5]
    env._known_cards["player_2"]["player_0"][2] = 1.0
    env._known_cards["player_2"]["player_1"][5] = 1.0
    env._min_kept_card["player_0"] = 2
    env._min_kept_card["player_1"] = 5
    env.step(700 + 1 * 10)
    assert env._hands["player_0"] == [5]
    assert 2 in env._hands["player_1"]
    assert known_indices(env, "player_0", "player_1") == [2]
    assert known_indices(env, "player_1", "player_0") == [5]
    assert known_indices(env, "player_2", "player_0") == [5]
    assert known_indices(env, "player_2", "player_1") == [2]
    assert env._min_kept_card["player_0"] == 5


def test_countess_forced_with_prince_or_king() -> None:
    env = fresh_env()
    env._hands["player_0"] = [8, 5]
    mask = env.observe("player_0")["action_mask"]
    valid = np.where(mask == 1)[0].tolist()
    assert valid == [890]


def test_spy_bonus_only_unique_alive_spy_player() -> None:
    env = fresh_env()
    env._hands["player_0"] = [4]
    env._hands["player_1"] = [3]
    env._played_cards["player_0"] = [0]
    env._played_cards["player_1"] = []
    env.terminations["player_2"] = True
    env.terminations["player_3"] = True
    env._resolve_round(["player_0", "player_1"])
    assert env._round_spy_winner == "player_0"
    assert env.rewards["player_0"] >= 1.0


def main() -> None:
    tests = [
        test_guard_wrong_does_not_reveal,
        test_priest_is_private_exact_info,
        test_baron_win_public_lower_bound_not_exact_card,
        test_baron_loss_public_lower_bound_not_exact_card,
        test_baron_tie_private_exact_info_for_both_players,
        test_handmaid_prevents_targeting_even_if_action_is_forced_in,
        test_prince_self_when_all_opponents_protected_can_discard_princess,
        test_chancellor_can_bottom_princess_without_elimination,
        test_king_is_private_bilateral_info,
        test_countess_forced_with_prince_or_king,
        test_spy_bonus_only_unique_alive_spy_player,
    ]
    for test in tests:
        test()
        print(f"ok - {test.__name__}")


if __name__ == "__main__":
    main()
