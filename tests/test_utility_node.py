"""Tests for :mod:`decisionpy.graph.utility_node`."""

from typing import Any

import pytest

from decisionpy.graph.node import Consistency, NodeKind
from decisionpy.graph.utility_node import UtilityNode

# --- helpers ----------------------------------------------------------------


def _values_taking_gain(gain: Any) -> float:  # noqa: ARG001
    """A `values` factory accepting exactly one kwarg: ``gain``."""
    return 1.0


def _values_matching(**_: Any) -> float:
    """A `values` factory accepting any kwargs (matches any parent set)."""
    return 1.0


# --- construction -----------------------------------------------------------


def test_minimal_utility_is_unconfigured() -> None:
    u = UtilityNode(name="payoff")

    assert u.parents == ()
    assert u.values is None
    assert u.consistency is Consistency.UNCONFIGURED
    assert u.is_sink is True
    assert u.is_discrete is False
    assert u.kind is NodeKind.UTILITY


def test_utility_with_values_is_consistent() -> None:
    u = UtilityNode(name="payoff", parents=("gain",), values=_values_taking_gain)

    assert u.values is _values_taking_gain
    assert u.consistency is Consistency.CONSISTENT


def test_utility_has_no_dist_or_states() -> None:
    """A utility carries a `values` function, not a dist or states."""
    u = UtilityNode(name="u")
    assert not hasattr(u, "dist")
    assert not hasattr(u, "states")


# --- field validation -------------------------------------------------------


def test_non_callable_values_rejected() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        UtilityNode(name="u", values=42)  # ty: ignore[invalid-argument-type]


def test_assigning_non_callable_values_raises() -> None:
    u = UtilityNode(name="u")
    with pytest.raises(TypeError, match="must be callable"):
        u.values = 3.14  # ty: ignore[invalid-assignment]


def test_assigning_none_values_is_allowed() -> None:
    """Setting values back to None returns the node to UNCONFIGURED."""
    u = UtilityNode(name="u", values=_values_matching)
    assert u.consistency is Consistency.CONSISTENT

    u.values = None
    assert u.consistency is Consistency.UNCONFIGURED


# --- consistency lifecycle --------------------------------------------------


def test_kwargs_values_matches_any_parents() -> None:
    u = UtilityNode(name="u", parents=("a", "b"), values=_values_matching)
    assert u.consistency is Consistency.CONSISTENT


def test_adding_parent_after_values_makes_utility_stale() -> None:
    u = UtilityNode(name="u", parents=("gain",), values=_values_taking_gain)
    assert u.consistency is Consistency.CONSISTENT

    u.add_parent("cost")
    assert u.consistency is Consistency.STALE


def test_values_with_extra_param_is_stale() -> None:
    def values_taking_two(gain: Any, cost: Any) -> float:  # noqa: ARG001
        return 1.0

    u = UtilityNode(name="u", parents=("gain",), values=values_taking_two)
    assert u.consistency is Consistency.STALE


# --- validate() gate --------------------------------------------------------


def test_validate_raises_on_unconfigured() -> None:
    with pytest.raises(ValueError, match="`values`"):
        UtilityNode(name="u").validate()


def test_validate_raises_on_stale() -> None:
    u = UtilityNode(name="u", parents=("gain",), values=_values_taking_gain)
    u.add_parent("cost")
    with pytest.raises(ValueError, match="does not match"):
        u.validate()


def test_validate_silent_on_consistent() -> None:
    UtilityNode(name="u", values=_values_matching).validate()  # no exception
