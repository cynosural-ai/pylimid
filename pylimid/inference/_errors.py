"""Errors shared by inference entry points and backends."""


class InferenceError(Exception):
    """An inference request cannot be handled by the given diagram or policy."""
