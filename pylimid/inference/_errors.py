"""Errors shared by inference entry points and backends."""


class InferenceError(Exception):
    """An inference or solving request fails for the given diagram or policy."""
