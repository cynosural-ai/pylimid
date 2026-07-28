"""Tests for :mod:`decisionpy.graph.decision_node`."""

import pytest

from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.node import Consistency, NodeKind

# --- construction -----------------------------------------------------------


def test_minimal_decision_is_unconfigured() -> None:
    d = DecisionNode(name="invest")

    assert d.parents == ()
    assert d.states is None
    assert d.consistency is Consistency.UNCONFIGURED
    assert d.is_discrete is False
    assert d.kind is NodeKind.DECISION
    assert d.is_sink is False


def test_decision_with_states_is_consistent() -> None:
    d = DecisionNode(name="invest", states=("buy", "hold", "sell"))

    assert d.states == ("buy", "hold", "sell")
    assert d.consistency is Consistency.CONSISTENT
    assert d.is_discrete is True


def test_decision_has_no_dist_field() -> None:
    """A decision carries actions, not a distribution."""
    assert not hasattr(DecisionNode(name="d"), "dist")


# --- field validation -------------------------------------------------------


def test_empty_states_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DecisionNode(name="d", states=())


def test_duplicate_states_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        DecisionNode(name="d", states=("a", "a"))


def test_non_tuple_states_rejected() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        DecisionNode(name="d", states=["a", "b"])  # type: ignore[arg-type]


def test_assigning_bad_states_raises() -> None:
    d = DecisionNode(name="d")
    with pytest.raises(ValueError, match="unique"):
        d.states = ("a", "a")


# --- consistency lifecycle --------------------------------------------------


def test_setting_states_makes_decision_consistent() -> None:
    d = DecisionNode(name="d")
    assert d.consistency is Consistency.UNCONFIGURED

    d.states = ("buy", "sell")
    assert d.consistency is Consistency.CONSISTENT


def test_decision_has_no_stale_state() -> None:
    """Adding an information parent never invalidates a decision (no STALE)."""
    d = DecisionNode(name="d", states=("buy", "sell"))
    assert d.consistency is Consistency.CONSISTENT

    d.add_parent("market")  # information set grows — decision stays consistent
    assert d.consistency is Consistency.CONSISTENT


# --- validate() gate --------------------------------------------------------


def test_validate_raises_on_unconfigured() -> None:
    with pytest.raises(ValueError, match="action space"):
        DecisionNode(name="d").validate()


def test_validate_silent_on_consistent() -> None:
    DecisionNode(name="d", states=("buy", "sell")).validate()  # no exception
