"""Central project paths used by scripts and apps."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


def data_path(filename: str) -> Path:
    return DATA_DIR / filename


def checkpoint_path(filename: str) -> Path:
    return CHECKPOINT_DIR / filename
