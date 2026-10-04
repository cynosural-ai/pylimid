"""
Tests for the shared :class:`~pylimid.graph.node.Node` base.

The base owns the contract every node type inherits — shared ``name`` /
``parents`` field validation and the consistency gate machinery. These tests
assert that inheritance: each subclass reuses the base's checks without
re-implementing them.
"""

from typing import Any

import pytest

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.node import Consistency, NodeKind
from pylimid.graph.utility_node import UtilityNode

# --- shared field validation is inherited -----------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: ChanceNode(name=""),
        lambda: DecisionNode(name=""),
        lambda: UtilityNode(name=""),
    ],
)
def test_empty_name_rejected_on_every_node_type(make: Any) -> None:
    """The base's `_validate_name` runs for all subclasses."""
    with pytest.raises(ValueError, match="`name`"):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: ChanceNode(name="x", parents=("x",)),
        lambda: DecisionNode(name="x", parents=("x",)),
        lambda: UtilityNode(name="x", parents=("x",)),
    ],
)
def test_self_parent_rejected_on_every_node_type(make: Any) -> None:
    """The base's `_validate_parents` (self-parent rule) runs for all subclasses."""
    with pytest.raises(ValueError, match="cannot list itself"):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: ChanceNode(name="y", parents=["a"]),  # ty: ignore[invalid-argument-type]
        lambda: DecisionNode(name="y", parents=["a"]),  # ty: ignore[invalid-argument-type]
        lambda: UtilityNode(name="y", parents=["a"]),  # ty: ignore[invalid-argument-type]
    ],
)
def test_non_tuple_parents_rejected_on_every_node_type(make: Any) -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        make()


def test_assignment_validation_inherited() -> None:
    """`__setattr__` validation runs post-construction for every subclass."""
    for node in (ChanceNode(name="x"), DecisionNode(name="x"), UtilityNode(name="x")):
        with pytest.raises(ValueError, match="`name`"):
            node.name = "  "


# --- mutation helpers inherited ---------------------------------------------


def test_add_remove_parent_inherited() -> None:
    for node in (ChanceNode(name="x"), DecisionNode(name="x"), UtilityNode(name="x")):
        node.add_parent("a")
        node.add_parent("a")  # idempotent
        node.add_parent("b")
        assert node.parents == ("a", "b")
        node.remove_parent("a")
        assert node.parents == ("b",)


# --- kind / is_sink defaults ------------------------------------------------


def test_each_node_type_reports_its_kind() -> None:
    assert ChanceNode(name="c").kind is NodeKind.CHANCE
    assert DecisionNode(name="d").kind is NodeKind.DECISION
    assert UtilityNode(name="u").kind is NodeKind.UTILITY


def test_only_utility_is_a_sink() -> None:
    """`is_sink` is the structural property the diagram's edge check reads."""
    assert ChanceNode(name="c").is_sink is False
    assert DecisionNode(name="d").is_sink is False
    assert UtilityNode(name="u").is_sink is True


# --- validate() gate is shared ----------------------------------------------


def test_validate_raises_on_inconsistent_for_all_types() -> None:
    """A fresh node of each type is UNCONFIGURED; validate() raises."""
    assert ChanceNode(name="c").consistency is Consistency.UNCONFIGURED
    assert DecisionNode(name="d").consistency is Consistency.UNCONFIGURED
    assert UtilityNode(name="u").consistency is Consistency.UNCONFIGURED

    with pytest.raises(ValueError, match="`dist`"):
        ChanceNode(name="c").validate()
    with pytest.raises(ValueError, match="action space"):
        DecisionNode(name="d").validate()
    with pytest.raises(ValueError, match="`values`"):
        UtilityNode(name="u").validate()


def test_consistency_message_belongs_to_the_node() -> None:
    """Each node type words its own non-CONSISTENT message."""
    assert "`dist`" in ChanceNode(name="c").consistency_message(
        Consistency.UNCONFIGURED
    )
    assert "action space" in DecisionNode(name="d").consistency_message(
        Consistency.UNCONFIGURED
    )
    assert "`values`" in UtilityNode(name="u").consistency_message(
        Consistency.UNCONFIGURED
    )
