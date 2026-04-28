"""Step5 Chancellor execution head.

This head is intentionally local: it does not decide whether to play
Chancellor. It only scores the legal Chancellor sub-actions once the model is
already in the Chancellor choice state.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import OBS_DIM


CHOICE_FEATURE_DIM = 62


def decode_chancellor_choice(pool: list[int], action: int) -> dict:
    pool = [int(card) for card in pool]
    action_idx = max(0, int(action) - 900)
    pool_size = len(pool)
    if pool_size >= 3:
        keep_idx = action_idx // 2
        order_idx = action_idx % 2
    elif pool_size == 2:
        keep_idx = action_idx
        order_idx = 0
    else:
        keep_idx = 0
        order_idx = 0
    if keep_idx < 0 or keep_idx >= max(1, pool_size):
        keep_idx = 0
    kept = pool[keep_idx] if pool else 0
    returned = list(pool)
    if pool:
        returned.pop(keep_idx)
    if order_idx == 1:
        returned.reverse()
    while len(returned) < 2:
        returned.append(0)
    return {
        "action": int(action),
        "pool": pool,
        "pool_size": int(pool_size),
        "keep_idx": int(keep_idx),
        "order_idx": int(order_idx),
        "kept_card": int(kept),
        "returned_cards": [int(returned[0]), int(returned[1])],
    }


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < size:
        vec[int(index)] = 1.0
    return vec


def chancellor_choice_features(pool: list[int], action: int, deck_remaining: int) -> np.ndarray:
    decoded = decode_chancellor_choice(pool, action)
    pool_cards = decoded["pool"]
    kept = decoded["kept_card"]
    returned = decoded["returned_cards"]
    pool_size = max(1, decoded["pool_size"])
    max_card = max(pool_cards) if pool_cards else kept
    min_card = min(pool_cards) if pool_cards else kept
    returned_arr = np.asarray(returned, dtype=np.float32)

    pool_counts = np.zeros(10, dtype=np.float32)
    for card in pool_cards:
        if 0 <= int(card) < 10:
            pool_counts[int(card)] += 1.0 / pool_size

    scalars = np.asarray(
        [
            decoded["pool_size"] / 3.0,
            decoded["keep_idx"] / 2.0,
            decoded["order_idx"],
            kept / 9.0,
            returned[0] / 9.0,
            returned[1] / 9.0,
            float(kept == max_card),
            float(kept == min_card),
            float(kept == 9),
            float(kept >= 7),
            float(any(card == 9 for card in returned)),
            float(any(card >= 7 for card in returned)),
            float(returned_arr.mean() / 9.0),
            float(returned_arr.max() / 9.0),
            float(returned_arr.min() / 9.0),
            float(deck_remaining / 21.0),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            _one_hot(int(action) - 900, 6),
            _one_hot(kept, 10),
            _one_hot(returned[0], 10),
            _one_hot(returned[1], 10),
            pool_counts,
            scalars,
        ]
    ).astype(np.float32)
    if features.shape[0] != CHOICE_FEATURE_DIM:
        raise ValueError(f"Bad Chancellor feature size: {features.shape[0]}")
    return features


class ChancellorExecutionHead(nn.Module):
    def __init__(self, hidden_dim: int = 192, dropout: float = 0.06):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + CHOICE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, choice_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, choice_features], dim=-1)).squeeze(-1)


def score_chancellor_actions(
    head: ChancellorExecutionHead,
    observation: np.ndarray,
    pool: list[int],
    deck_remaining: int,
    actions: list[int],
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    obs = np.asarray(observation, dtype=np.float32)
    feats = np.stack([chancellor_choice_features(pool, action, deck_remaining) for action in actions])
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).expand(len(actions), -1)
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = head.to(device).eval()(obs_t, feats_t)
    return scores.detach().cpu().numpy().astype(np.float32)


def load_chancellor_head(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "step5_chancellor_execution_head":
        raise ValueError(f"{path} is not a Step5 Chancellor execution checkpoint")
    head = ChancellorExecutionHead(
        hidden_dim=int(checkpoint.get("hidden_dim", 192)),
        dropout=float(checkpoint.get("dropout", 0.06)),
    )
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()
    return head, checkpoint
