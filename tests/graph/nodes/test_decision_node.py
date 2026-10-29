"""Tests for :mod:`pylimid.graph.nodes.decision_node`."""

import pytest

from pylimid.graph import Interval, Real
from pylimid.graph.nodes.decision_node import DecisionNode
from pylimid.graph.nodes.node import Consistency, NodeKind

# --- construction -----------------------------------------------------------


def test_minimal_decision_is_unconfigured() -> None:
    d = DecisionNode(name="invest")

    assert d.parents == ()
    assert d.states is None
    assert d.action_space is None
    assert d.consistency is Consistency.UNCONFIGURED
    assert d.is_categorical is False
    assert d.kind is NodeKind.DECISION
    assert d.is_sink is False


def test_decision_with_states_is_consistent() -> None:
    d = DecisionNode(name="invest", states=("buy", "hold", "sell"))

    assert d.states == ("buy", "hold", "sell")
    assert d.consistency is Consistency.CONSISTENT
    assert d.is_categorical is True


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
        DecisionNode(name="d", states=["a", "b"])  # ty: ignore[invalid-argument-type]


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


def test_information_parent_changes_preserve_categorical_decision_consistency() -> None:
    """Adding an information parent never invalidates the action declaration."""
    d = DecisionNode(name="d", states=("buy", "sell"))
    assert d.consistency is Consistency.CONSISTENT

    d.add_parent("market")  # information set grows — decision stays consistent
    assert d.consistency is Consistency.CONSISTENT


# --- validate() gate --------------------------------------------------------


def test_validate_silent_on_consistent() -> None:
    DecisionNode(name="d", states=("buy", "sell")).validate()  # no exception


@pytest.mark.parametrize("domain", [Real(), Interval(0.0, 100.0)])
def test_continuous_decision_is_configured_without_named_states(domain):
    decision = DecisionNode("power", action_space=domain)
    assert decision.consistency is Consistency.CONSISTENT
    assert decision.states is None
    assert not decision.is_categorical
    assert decision.policy_family is None
    decision.validate()
    decision.add_parent("temperature")
    decision.validate()
    decision.remove_parent("temperature")
    decision.validate()


@pytest.mark.parametrize("domain", [Real(), Interval(0.0, 100.0)])
def test_categorical_and_continuous_declarations_conflict_at_validation(domain):
    decision = DecisionNode("power", states=("off", "on"), action_space=domain)
    assert decision.consistency is Consistency.STALE
    with pytest.raises(ValueError, match="both `states` and `action_space`"):
        decision.validate()


def test_action_domain_can_be_edited_through_conflicting_and_incomplete_states():
    decision = DecisionNode("power")
    decision.action_space = Real()
    assert decision.consistency is Consistency.CONSISTENT
    decision.states = ("off", "on")
    assert decision.consistency is Consistency.STALE
    decision.action_space = None
    assert decision.consistency is Consistency.CONSISTENT
    assert decision.is_categorical
    decision.states = None
    assert decision.consistency is Consistency.UNCONFIGURED
    decision.action_space = Interval(0.0, 100.0)
    assert decision.consistency is Consistency.CONSISTENT


@pytest.mark.parametrize("value", ["real", (0.0, 100.0), 1.0, Real])
def test_invalid_action_space_is_rejected_on_construction_and_assignment(value):
    with pytest.raises(TypeError, match="`action_space` must be"):
        DecisionNode("power", action_space=value)
    decision = DecisionNode("power", action_space=Real())
    with pytest.raises(TypeError, match="`action_space` must be"):
        decision.action_space = value
    assert decision.action_space == Real()


def test_family_assignment_rejects_unsupported_representations():
    with pytest.raises(TypeError, match="policy_family"):
        DecisionNode("action", states=("off", "on"), policy_family="network")  # ty: ignore[invalid-argument-type]
