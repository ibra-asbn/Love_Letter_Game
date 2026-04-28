"""Belief-aware inference policy for existing Love Letter checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefConditionedPolicy,
    BeliefHead as BeliefConditionedHead,
)

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_PLAYERS = 4
NUM_OPPONENTS = NUM_PLAYERS - 1


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(LATENT, LATENT)

    def forward(self, x, h_in):
        features = self.feature_extractor(x)
        return features, self.rnn(features, h_in)


class MaskedActor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, ACTION_DIM)


class BeliefHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = NUM_OPPONENTS
        self.head = nn.Linear(LATENT, NUM_OPPONENTS * 10)


@dataclass
class BeliefDebug:
    probs: np.ndarray | None
    used_belief: bool


class BeliefAwarePolicy:
    """
    Wraps the existing actor and optional belief head.

    Existing checkpoints were trained with an auxiliary belief head, but the old
    inference path ignored it. This wrapper keeps the actor's raw preference and
    nudges valid action logits using the belief probabilities at every decision.
    """

    def __init__(
        self,
        encoder: RecurrentEncoder,
        actor: MaskedActor,
        belief_head: BeliefHead | None = None,
        belief_strength: float = 1.0,
    ):
        self.encoder = encoder
        self.actor = actor
        self.belief_head = belief_head
        self.belief_strength = belief_strength
        self.last_debug = BeliefDebug(probs=None, used_belief=False)

    def eval(self):
        self.encoder.eval()
        self.actor.eval()
        if self.belief_head is not None:
            self.belief_head.eval()
        return self

    def act(self, obs_dict, hidden_state=None, agent_id="player_0"):
        with torch.no_grad():
            x = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
            h_in = hidden_state if hidden_state is not None else torch.zeros(1, LATENT)

            _features, new_state = self.encoder(x, h_in)
            logits = self.actor.head(new_state)

            belief_probs = None
            if self.belief_head is not None:
                belief_logits = self.belief_head.head(new_state).view(1, NUM_OPPONENTS, 10)
                belief_probs = torch.softmax(belief_logits, dim=-1)
                logits = self._apply_belief_to_logits(logits, mask, belief_probs, obs_dict, agent_id)

            logits = logits.masked_fill(~mask, -1e9)
            action = int(logits.argmax(dim=-1).item())

            self.last_debug = BeliefDebug(
                probs=belief_probs.squeeze(0).cpu().numpy() if belief_probs is not None else None,
                used_belief=belief_probs is not None,
            )
            return action, new_state

    def _apply_belief_to_logits(self, logits, mask, belief_probs, obs_dict, agent_id):
        adjusted = logits.clone()
        valid_actions = torch.where(mask.squeeze(0))[0].tolist()
        if not valid_actions:
            return adjusted

        my_idx = int(agent_id.rsplit("_", 1)[1])
        my_cards = _cards_from_obs_hand(obs_dict["observation"])

        for action in valid_actions:
            action = int(action)
            if action >= 900:
                continue

            card = action // 100
            target_idx = (action % 100) // 10
            guess = action % 10
            target_dim = _target_belief_dim(my_idx, target_idx)
            probs = belief_probs[0, target_dim] if target_dim is not None else None
            kept = _kept_card_after_play(my_cards, card)

            delta = self._belief_delta(card, guess, probs, kept, target_idx == my_idx)
            adjusted[0, action] += self.belief_strength * delta

        return adjusted

    @staticmethod
    def _belief_delta(card, guess, probs, kept, targets_self):
        if card == 1 and probs is not None:
            # Garde: prefer high-probability guesses and heavily reward the top guess.
            best_guess = _best_guard_guess(probs)
            prob_guess = probs[guess]
            bonus = 5.0 * prob_guess
            if guess == best_guess:
                bonus += 1.5
            return bonus

        if card == 3 and probs is not None and kept is not None:
            # Baron: compare the card we keep with the target's belief distribution.
            p_lower = probs[:kept].sum() if kept > 0 else torch.tensor(0.0)
            p_equal = probs[kept]
            p_higher = probs[kept + 1 :].sum() if kept < 9 else torch.tensor(0.0)
            return 5.0 * p_lower - 7.0 * p_higher - 1.0 * p_equal

        if card == 5:
            if targets_self:
                if kept == 9:
                    return torch.tensor(-8.0)
                if kept is not None and kept <= 2:
                    return torch.tensor(3.0)
                if kept is not None and kept <= 4:
                    return torch.tensor(1.0)
                return torch.tensor(-0.5)
            if probs is not None:
                # Prince: strongest when belief says Princess/Countess/high cards.
                return 7.0 * probs[9] + 2.5 * probs[8] + 1.0 * probs[7]

        if card == 7 and probs is not None and kept is not None:
            # King: exchange is attractive only if target's expected card is better.
            expected_target = torch.sum(probs * torch.arange(10, dtype=probs.dtype, device=probs.device))
            return 3.0 * ((expected_target - kept) / 9.0)

        if card == 2 and probs is not None:
            # Priest: prefer targets where belief is uncertain.
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum() / np.log(10)
            return 1.0 * entropy

        return torch.tensor(0.0, dtype=logits_dtype(probs))


def logits_dtype(probs):
    return probs.dtype if probs is not None else torch.float32


def load_belief_policy(path: str | Path, belief_strength: float = 1.0):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") == "belief_conditioned_actor_v1":
        encoder = BeliefConditionedEncoder()
        belief_head = BeliefConditionedHead()
        actor = BeliefConditionedActor()
        encoder.load_state_dict(ckpt["encoder"])
        belief_head.load_state_dict(ckpt["belief_head"])
        actor.load_state_dict(ckpt["actor"])
        return BeliefConditionedPolicy(encoder, belief_head, actor).eval()

    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])

    belief_head = None
    if "belief_head" in ckpt:
        belief_head = BeliefHead(encoder)
        belief_head.load_state_dict(ckpt["belief_head"])

    policy = BeliefAwarePolicy(encoder, actor, belief_head, belief_strength=belief_strength)
    return policy.eval()


def _cards_from_obs_hand(obs):
    hand_block = np.asarray(obs[:10], dtype=np.float32)
    cards = []
    for card, value in enumerate(hand_block):
        count = int(round(float(value) * 3.0))
        cards.extend([card] * max(0, count))
    return cards


def _kept_card_after_play(hand, played_card):
    remaining = list(hand)
    if played_card in remaining:
        remaining.remove(played_card)
    return remaining[0] if remaining else None


def _target_belief_dim(my_idx, target_idx):
    if target_idx >= NUM_PLAYERS or target_idx == my_idx:
        return None
    rel = (target_idx - my_idx) % NUM_PLAYERS
    if rel == 0:
        return None
    return rel - 1


def _best_guard_guess(probs):
    guard_probs = probs.clone()
    guard_probs[1] = -1.0
    return int(torch.argmax(guard_probs).item())
