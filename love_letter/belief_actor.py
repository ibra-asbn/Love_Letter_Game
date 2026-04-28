"""Actor architecture that learns to consume belief probabilities directly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_OPPONENTS = 3
NUM_CARDS = 10
BELIEF_DIM = NUM_OPPONENTS * NUM_CARDS


class BeliefConditionedEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(latent, latent)
        self.hidden_size = latent

    def forward_hidden(self, obs_t, hidden_state=None):
        features = self.feature_extractor(obs_t)
        if hidden_state is None:
            hidden_state = torch.zeros(
                obs_t.shape[0],
                self.hidden_size,
                dtype=obs_t.dtype,
                device=obs_t.device,
            )
        return self.rnn(features, hidden_state)


class BeliefHead(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.head = nn.Linear(latent, BELIEF_DIM)

    def forward(self, hidden):
        logits = self.head(hidden).view(-1, NUM_OPPONENTS, NUM_CARDS)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class BeliefConditionedActor(nn.Module):
    def __init__(self, latent=LATENT, belief_dim=BELIEF_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + belief_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim),
        )

    def forward(self, hidden, belief_probs, action_mask=None):
        belief_flat = belief_probs.reshape(hidden.shape[0], -1)
        logits = self.net(torch.cat([hidden, belief_flat], dim=-1))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


@dataclass
class BeliefConditionedDebug:
    belief_probs: torch.Tensor | None = None


class BeliefConditionedPolicy:
    def __init__(self, encoder, belief_head, actor):
        self.encoder = encoder
        self.belief_head = belief_head
        self.actor = actor
        self.last_debug = BeliefConditionedDebug()

    def eval(self):
        self.encoder.eval()
        self.belief_head.eval()
        self.actor.eval()
        return self

    def act(self, obs_dict, hidden_state=None, agent_id=None):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
            hidden = self.encoder.forward_hidden(obs_t, hidden_state)
            _belief_logits, belief_probs = self.belief_head(hidden)
            logits = self.actor(hidden, belief_probs, mask_t)
            action = int(logits.argmax(dim=-1).item())
            self.last_debug = BeliefConditionedDebug(belief_probs=belief_probs.detach().cpu())
        return action, hidden


def load_belief_conditioned_policy(path: str | Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = BeliefConditionedEncoder()
    belief_head = BeliefHead()
    actor = BeliefConditionedActor()
    encoder.load_state_dict(ckpt["encoder"])
    belief_head.load_state_dict(ckpt["belief_head"])
    actor.load_state_dict(ckpt["actor"])
    return BeliefConditionedPolicy(encoder, belief_head, actor).eval()
