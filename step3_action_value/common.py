"""Shared Step3 utilities.

This module keeps the active Step3 code independent from archived experiments.
Older attempts still exist for traceability, but the current advantage/DAgger
pipeline should import reusable helpers from here.
"""

from __future__ import annotations

import numpy as np
import torch

from love_letter.belief_policy import load_belief_policy
from step2_rl_finetune.evaluate_step2 import random_action
from step3_action_value.mini_rollout_probe import choose_actions_for_probe


def action_parts(actions: torch.Tensor):
    actions = actions.long()
    is_choice = (actions >= 900) & (actions <= 905)
    card = torch.div(actions, 100, rounding_mode="floor").clamp(0, 9)
    target = torch.div(actions % 100, 10, rounding_mode="floor").clamp(0, 9)
    guess = (actions % 10).clamp(0, 9)
    card = torch.where(is_choice, torch.full_like(card, 10), card)
    target = torch.where(is_choice, (actions - 900).clamp(0, 9), target)
    guess = torch.where(is_choice, torch.zeros_like(guess), guess)
    return card, target, guess


def candidate_actions(env, model_action: int, heuristic_action: int, max_actions: int) -> list[int]:
    """Use one consistent candidate generator for Step3 training and inference."""
    candidates = choose_actions_for_probe(env, max_actions)
    for forced in [model_action, heuristic_action]:
        if forced not in candidates:
            candidates = [forced] + candidates
    return list(dict.fromkeys(int(action) for action in candidates))[:max_actions]


class Player0FeaturePolicy:
    """Wrapper around a belief actor that exposes latent and belief features."""

    def __init__(self, checkpoint):
        self.policy = load_belief_policy(checkpoint)
        self.state = None

    def act(self, obs_dict):
        action, self.state = self.policy.act(obs_dict, self.state, agent_id="player_0")
        hidden = self.state.detach().cpu().squeeze(0).numpy()
        debug = getattr(self.policy, "last_debug", None)
        belief = _debug_belief_array(debug)
        if belief is None:
            belief = np.zeros((3, 10), dtype=np.float32)
        return int(action), hidden.astype(np.float32), belief.astype(np.float32)


def _debug_belief_array(debug):
    if debug is None:
        return None
    belief = getattr(debug, "belief_probs", None)
    if belief is None:
        belief = getattr(debug, "probs", None)
    if belief is None:
        return None
    if isinstance(belief, torch.Tensor):
        return belief.detach().cpu().squeeze(0).numpy()
    return np.asarray(belief, dtype=np.float32)


def opponent_action(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)
