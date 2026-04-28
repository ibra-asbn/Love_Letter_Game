"""Core package for the Love Letter reinforcement-learning project."""

__all__ = ["LoveLetterRLEnv"]


def __getattr__(name):
    if name == "LoveLetterRLEnv":
        from .engine import LoveLetterRLEnv

        return LoveLetterRLEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
