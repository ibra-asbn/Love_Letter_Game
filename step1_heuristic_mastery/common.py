"""Shared utilities for the heuristic-mastery step."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STEP_DIR = Path(__file__).resolve().parent
STEP_DATA_DIR = STEP_DIR / "data"
STEP_CHECKPOINT_DIR = STEP_DIR / "checkpoints"
STEP_LOG_DIR = STEP_DIR / "logs"
STEP_REPORT_DIR = STEP_DIR / "reports"
PARIS_TZ = ZoneInfo("Europe/Paris")

CARD_NAMES = [
    "Espionne",
    "Garde",
    "Pretre",
    "Baron",
    "Servante",
    "Prince",
    "Chancelier",
    "Roi",
    "Comtesse",
    "Princesse",
]


def ensure_step_dirs():
    for path in [STEP_DATA_DIR, STEP_CHECKPOINT_DIR, STEP_LOG_DIR, STEP_REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def now_stamp():
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


class ExperimentLogger:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self):
        if self.path:
            self.path.write_text("", encoding="utf-8")

    def write(self, title, expected=None, actual=None, details=None):
        message = f"[{now_stamp()}] {title}"
        print(message, flush=True)
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"## {message}\n\n")
            if expected is not None:
                f.write(f"**Attendu**: {expected}\n\n")
            if actual is not None:
                f.write(f"**Obtenu**: {actual}\n\n")
            if details is not None:
                f.write("```json\n")
                f.write(json.dumps(details, indent=2, ensure_ascii=False))
                f.write("\n```\n\n")


def resolve_step_path(path, base_dir):
    path = Path(path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return base_dir / path


def decode_action(action):
    if action >= 900:
        return {"kind": "chancellor_choice", "card_name": "ChancellorChoice", "action": int(action)}
    card = int(action) // 100
    target = (int(action) % 100) // 10
    guess = int(action) % 10
    return {
        "kind": "card",
        "card": card,
        "card_name": CARD_NAMES[card] if 0 <= card < len(CARD_NAMES) else str(card),
        "target": int(target),
        "guess": int(guess),
        "guess_name": CARD_NAMES[guess] if 0 <= guess < len(CARD_NAMES) else str(guess),
        "action": int(action),
    }


def absolute_to_relative_action(action, my_idx):
    """Map env absolute target indices into the policy's relative target space."""
    action = int(action)
    if action >= 900:
        return action
    card = action // 100
    target = (action % 100) // 10
    guess = action % 10
    if target == 9:
        rel_target = 9
    else:
        rel_target = (target - my_idx) % 4
    return int(card * 100 + rel_target * 10 + guess)


def relative_to_absolute_action(action, my_idx):
    """Map a relative policy action back into env absolute target indices."""
    action = int(action)
    if action >= 900:
        return action
    card = action // 100
    target = (action % 100) // 10
    guess = action % 10
    if target == 9:
        abs_target = 9
    else:
        abs_target = (my_idx + target) % 4
    return int(card * 100 + abs_target * 10 + guess)


def absolute_to_relative_mask(mask, my_idx):
    """Convert an env action mask to the relative action space used by the student."""
    import numpy as np

    relative = np.zeros_like(mask)
    for action in np.where(mask == 1)[0]:
        relative[absolute_to_relative_action(int(action), my_idx)] = 1
    return relative


def composite_score(configs):
    weights = {"vs_0H_3R": 0.10, "vs_1H_2R": 0.20, "vs_2H_1R": 0.30, "vs_3H": 0.40}
    return float(sum(weights[name] * configs[name]["winrate"] for name in weights))
