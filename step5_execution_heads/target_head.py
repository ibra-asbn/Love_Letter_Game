"""Generic Step5 target-execution head.

This head is local: it does not decide whether to play Priest, Baron or King.
It only scores legal target choices after Step3 has already selected that card.
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


MAX_CARD_COUNTS = np.asarray([2, 6, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.float32)
TARGET_FEATURE_DIM = 90

CARD_TO_KIND = {
    2: "priest_target",
    3: "baron_target",
    7: "king_target",
}

KIND_TO_CARD = {
    "priest_target": 2,
    "baron_target": 3,
    "baron_low_target": 3,
    "king_target": 7,
}

KIND_TO_LABEL = {
    "priest_target": "Pretre",
    "baron_target": "Baron",
    "baron_low_target": "Baron faible",
    "king_target": "Roi",
}


def action_card(action: int) -> int:
    return int(action) // 100


def action_target(action: int) -> int:
    return (int(action) % 100) // 10


def seat_index(seat: str) -> int:
    try:
        return int(str(seat).rsplit("_", 1)[1])
    except Exception:
        return 0


def relative_target(seat: str, target: int) -> int:
    return (int(target) - seat_index(seat)) % 4


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < size:
        vec[int(index)] = 1.0
    return vec


def infer_kept_card(card: int, hand: list[int]) -> int:
    cards = [int(item) for item in hand]
    if int(card) in cards:
        remaining = list(cards)
        remaining.remove(int(card))
        if remaining:
            return int(remaining[0])
    return int(cards[0]) if cards else 0


def phase_one_hot(deck_remaining: int) -> np.ndarray:
    if deck_remaining >= 11:
        idx = 0
    elif deck_remaining >= 6:
        idx = 1
    else:
        idx = 2
    return _one_hot(idx, 3)


def target_distribution_from_obs(observation: np.ndarray | None, seat: str, target: int) -> dict:
    if observation is None:
        probs = np.ones(10, dtype=np.float32) / 10.0
        return {
            "probs": probs,
            "known_flag": 0.0,
            "public_min": 0.0,
            "unchanged": 0.0,
            "countess_voluntary": 0.0,
            "last_played": np.zeros(10, dtype=np.float32),
        }
    obs = np.asarray(observation, dtype=np.float32)
    rel = relative_target(seat, target)
    if rel == 0 or int(target) >= 4:
        probs = np.ones(10, dtype=np.float32) / 10.0
        return {
            "probs": probs,
            "known_flag": 0.0,
            "public_min": 0.0,
            "unchanged": 0.0,
            "countess_voluntary": 0.0,
            "last_played": np.zeros(10, dtype=np.float32),
        }
    rel_idx = rel - 1
    known = obs[81 + rel_idx * 10 : 81 + (rel_idx + 1) * 10].astype(np.float32)
    public_min = float(obs[155 + rel_idx] * 9.0) if len(obs) >= 158 else 0.0
    if known.sum() > 0:
        probs = known / known.sum()
        known_flag = 1.0
    else:
        counts = np.clip(obs[20:30].astype(np.float32) * MAX_CARD_COUNTS, 0.0, None)
        if public_min > 0.5:
            counts[: int(round(public_min))] = 0.0
        probs = counts / counts.sum() if counts.sum() > 0 else np.ones(10, dtype=np.float32) / 10.0
        known_flag = 0.0
    return {
        "probs": probs.astype(np.float32),
        "known_flag": float(known_flag),
        "public_min": float(public_min / 9.0),
        "unchanged": float(obs[119 + rel_idx]) if len(obs) >= 122 else 0.0,
        "countess_voluntary": float(obs[122 + rel_idx]) if len(obs) >= 125 else 0.0,
        "last_played": obs[36 + rel_idx * 10 : 36 + (rel_idx + 1) * 10].astype(np.float32),
    }


def target_action_features(
    action: int,
    seat: str,
    hand: list[int],
    deck_remaining: int,
    candidate_count: int = 0,
    observation: np.ndarray | None = None,
) -> np.ndarray:
    card = action_card(action)
    target = action_target(action)
    rel = relative_target(seat, target)
    kept = infer_kept_card(card, hand)
    hand_cards = [int(item) for item in hand]
    hand_min = min(hand_cards) if hand_cards else kept
    hand_max = max(hand_cards) if hand_cards else kept

    hand_counts = np.zeros(10, dtype=np.float32)
    for item in hand_cards:
        if 0 <= int(item) < 10:
            hand_counts[int(item)] += 1.0 / max(1, len(hand_cards))

    target_dist = target_distribution_from_obs(observation, seat, target)
    probs = target_dist["probs"]
    p_lower = float(probs[:kept].sum()) if kept > 0 else 0.0
    p_equal = float(probs[kept]) if 0 <= kept < 10 else 0.0
    p_higher = float(probs[kept + 1 :].sum()) if kept < 9 else 0.0
    expected = float(np.dot(probs, np.arange(10, dtype=np.float32)) / 9.0)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum() / np.log(10.0))

    scalars = np.asarray(
        [
            float(deck_remaining / 21.0),
            float(target / 3.0),
            float(rel / 3.0),
            float(kept / 9.0),
            float(hand_min / 9.0),
            float(hand_max / 9.0),
            float(9 in hand_cards),
            float(8 in hand_cards),
            float(kept <= 4),
            float(kept >= 7),
            float(target == seat_index(seat)),
            float(card == 2),
            float(card == 3),
            float(card == 7),
            float(candidate_count / 4.0),
            p_lower,
            p_equal,
            p_higher,
            expected,
            entropy,
            float(target_dist["known_flag"]),
            float(target_dist["public_min"]),
            float(target_dist["unchanged"]),
            float(target_dist["countess_voluntary"]),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            _one_hot(card, 10),
            _one_hot(target, 4),
            _one_hot(rel, 4),
            hand_counts,
            _one_hot(kept, 10),
            phase_one_hot(deck_remaining),
            scalars,
            _one_hot(int(action) % 10, 3),
            _one_hot(len(hand_cards), 2),
            probs,
            target_dist["last_played"],
        ]
    ).astype(np.float32)
    if features.shape[0] != TARGET_FEATURE_DIM:
        raise ValueError(f"Bad target feature size: {features.shape[0]}")
    return features


class TargetExecutionHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + TARGET_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, target_features], dim=-1)).squeeze(-1)


def score_target_actions(
    head: TargetExecutionHead,
    observation: np.ndarray,
    actions: list[int],
    seat: str,
    hand: list[int],
    deck_remaining: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    obs = np.asarray(observation, dtype=np.float32)
    feats = np.stack(
        [
            target_action_features(action, seat, hand, deck_remaining, len(actions), observation=observation)
            for action in actions
        ]
    )
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).expand(len(actions), -1)
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = head.to(device).eval()(obs_t, feats_t)
    return scores.detach().cpu().numpy().astype(np.float32)


def load_target_head(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "step5_target_execution_head":
        raise ValueError(f"{path} is not a Step5 target execution checkpoint")
    head = TargetExecutionHead(
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.10)),
    )
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()
    return head, checkpoint
