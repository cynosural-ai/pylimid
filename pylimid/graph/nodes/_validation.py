"""Field validation for node names, parents, labels, and configurable functions."""

from typing import cast


def validate_name(name: object) -> None:
    """
    Validate a node identifier.

    Args:
        name: Value assigned to the node's name.

    Raises:
        ValueError: If the name is not a non-empty string.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`name` must be a non-empty string, got {name!r}.")


def validate_parents(name: str, parents: object) -> None:
    """
    Validate declared parent names for one node.

    Args:
        name: The node's own validated name.
        parents: Value assigned to the node's parents.

    Raises:
        TypeError: If parents is not a tuple.
        ValueError: If a parent is blank, not a string, duplicated, or the node itself.
    """
    if not isinstance(parents, tuple):
        raise TypeError(
            f"`parents` for {name!r} must be a tuple, got {type(parents).__name__}."
        )
    if not all(isinstance(p, str) and p.strip() for p in parents):
        raise ValueError(
            f"`parents` for {name!r} must all be non-empty strings, got {parents!r}."
        )
    seen: set[str] = set()
    for parent in cast(tuple[str, ...], parents):
        if parent == name:
            raise ValueError(f"Node {name!r} cannot list itself among its parents.")
        if parent in seen:
            raise ValueError(f"`parents` for {name!r} contains duplicate {parent!r}.")
        seen.add(parent)


def validate_states(states: object) -> None:
    """
    Validate optional categorical state or action labels.

    Args:
        states: A tuple of unique, non-empty strings, or ``None`` when no labels are
            declared.

    Raises:
        TypeError: If the value is neither a tuple nor ``None``.
        ValueError: If the tuple is empty or contains duplicate, blank, or non-string
            labels.
    """
    if states is None:
        return
    if not isinstance(states, tuple):
        raise TypeError(
            f"`states` must be a tuple or None, got {type(states).__name__}."
        )
    if not states:
        raise ValueError(
            "`states`, when provided, must be non-empty; use `None` when no "
            "state labels are declared."
        )
    if not all(isinstance(state, str) and state.strip() for state in states):
        raise ValueError(f"`states` must all be non-empty strings, got {states!r}.")
    if len(set(states)) != len(states):
        raise ValueError(f"`states` must be unique, got {states!r}.")


def validate_dist(dist: object) -> None:
    """
    Validate an optional chance distribution function.

    Args:
        dist: Value assigned to a chance node's distribution function.

    Raises:
        TypeError: If the value is neither callable nor ``None``.
    """
    if dist is not None and not callable(dist):
        raise TypeError(
            f"`dist` must be callable or None, "
            f"got object of type {type(dist).__name__}."
        )


def validate_values(values: object) -> None:
    """
    Validate an optional utility function.

    Args:
        values: Value assigned to a utility node's payoff function.

    Raises:
        TypeError: If the value is neither callable nor ``None``.
    """
    if values is not None and not callable(values):
        raise TypeError(
            f"`values` must be callable or None, "
            f"got object of type {type(values).__name__}."
        )
