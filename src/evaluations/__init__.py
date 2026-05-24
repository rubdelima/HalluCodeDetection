from __future__ import annotations

__all__ = ["evaluate_models"]


def __getattr__(name: str):
    if name == "evaluate_models":
        from src.evaluations.run import evaluate_models

        return evaluate_models
    raise AttributeError(f"module 'src.evaluations' has no attribute '{name}'")