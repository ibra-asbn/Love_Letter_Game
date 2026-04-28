"""Shared helpers for step 2 RL fine-tuning."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STEP_DIR = PROJECT_ROOT / "step2_rl_finetune"
STEP_CHECKPOINT_DIR = STEP_DIR / "checkpoints"
STEP_LOG_DIR = STEP_DIR / "logs"
STEP_REPORT_DIR = STEP_DIR / "reports"

PARIS_TZ = ZoneInfo("Europe/Paris")


def now_stamp() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_step_dirs() -> None:
    for path in [STEP_CHECKPOINT_DIR, STEP_LOG_DIR, STEP_REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_step_path(name_or_path: str | Path, default_dir: Path) -> Path:
    path = Path(name_or_path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return default_dir / path


def resolve_checkpoint(name_or_path: str | Path) -> Path:
    """Resolve a checkpoint from step2, step1, models/checkpoints, or a direct path."""
    path = Path(name_or_path)
    candidates = [
        path,
        STEP_CHECKPOINT_DIR / path,
        PROJECT_ROOT / "step1_heuristic_mastery" / "checkpoints" / path,
        PROJECT_ROOT / "models" / "checkpoints" / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")


def composite_score(configs: dict) -> float:
    weights = {
        "vs_0H_3R": 0.10,
        "vs_1H_2R": 0.20,
        "vs_2H_1R": 0.30,
        "vs_3H": 0.40,
    }
    return float(sum(weights[name] * configs[name]["winrate"] for name in weights if name in configs))


def arena_summary(configs: dict) -> dict:
    return {
        name: {
            "winrate": round(float(values["winrate"]), 4),
            "mean_reward": round(float(values["mean_reward"]), 4),
            "ci95": round(float(values.get("winrate_ci95", 0.0)), 4),
        }
        for name, values in configs.items()
    }


class ExperimentLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.path:
            self.path.write_text("", encoding="utf-8")

    def write(self, title: str, expected=None, actual=None, details=None) -> None:
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
                if isinstance(details, (dict, list)):
                    f.write("```json\n")
                    f.write(json.dumps(details, indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                else:
                    f.write(f"{details}\n\n")

