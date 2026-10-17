"""Errors shared by posterior inference, policy handling, and solving."""


class InferenceError(Exception):
    """An inference or solving request fails for the given diagram or policy."""
