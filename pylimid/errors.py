"""Public errors for inference requests and policy optimization."""

__all__ = ["InferenceError", "SolveError"]


class InferenceError(Exception):
    """An inference request or supplied policy is incompatible with the diagram."""


class SolveError(Exception):
    """Policy optimization or final evaluation produced a non-finite result."""
