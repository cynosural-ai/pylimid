"""Tests for :mod:`decisionpy.graph.chance_node`."""

from typing import Any

import pytest

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.node import Consistency

# --- helpers ----------------------------------------------------------------


def _dist_matching(**_: Any) -> object:
    """A dist factory accepting any kwargs (matches any parent set)."""
    return object()


def _dist_taking_rain(rain: Any) -> object:  # noqa: ARG001
    """A dist factory whose signature accepts exactly one kwarg: ``rain``."""
    return object()


# --- construction -----------------------------------------------------------


def test_minimal_node_has_only_name() -> None:
    """A node built with just a name defaults to unconfigured, no parents."""
    node = ChanceNode(name="rain")

    assert node.name == "rain"
    assert node.parents == ()
    assert node.dist is None
    assert node.states is None
    assert node.consistency is Consistency.UNCONFIGURED
    assert node.is_discrete is False


def test_full_node_round_trips_fields() -> None:
    """A fully-specified node stores every field and is consistent."""
    node = ChanceNode(
        name="wet_grass",
        parents=("rain",),
        dist=_dist_taking_rain,
        states=("dry", "wet"),
    )

    assert node.parents == ("rain",)
    assert node.dist is _dist_taking_rain
    assert node.states == ("dry", "wet")
    assert node.consistency is Consistency.CONSISTENT
    assert node.is_discrete is True


# --- field validation on construction ---------------------------------------


@pytest.mark.parametrize("bad_name", ["", "  "])
def test_empty_or_blank_name_rejected(bad_name: str) -> None:
    with pytest.raises(ValueError, match="`name`"):
        ChanceNode(name=bad_name)


def test_non_string_name_rejected() -> None:
    with pytest.raises(ValueError, match="`name`"):
        ChanceNode(name=123)  # ty: ignore[invalid-argument-type]


def test_self_parent_rejected() -> None:
    with pytest.raises(ValueError, match="cannot list itself"):
        ChanceNode(name="x", parents=("x",))


def test_duplicate_parent_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ChanceNode(name="y", parents=("x", "x"))


def test_blank_parent_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ChanceNode(name="y", parents=("  ",))


def test_non_tuple_parents_rejected() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        ChanceNode(name="y", parents=["x"])  # ty: ignore[invalid-argument-type]


def test_non_callable_dist_rejected() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        ChanceNode(name="x", dist=42)  # ty: ignore[invalid-argument-type]


def test_empty_states_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ChanceNode(name="x", states=())


def test_duplicate_states_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        ChanceNode(name="x", states=("a", "a"))


def test_non_tuple_states_rejected() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        ChanceNode(name="x", states=["a"])  # ty: ignore[invalid-argument-type]


# --- mutability: field validation runs on assignment ------------------------


def test_node_is_mutable() -> None:
    """Fields can be assigned after construction."""
    node = ChanceNode(name="x")

    node.parents = ("y",)
    node.dist = _dist_taking_rain
    node.states = ("a", "b")

    assert node.parents == ("y",)
    assert node.dist is _dist_taking_rain
    assert node.states == ("a", "b")


def test_assigning_bad_name_raises() -> None:
    """Proves __setattr__ validates, not just __init__."""
    node = ChanceNode(name="x")
    with pytest.raises(ValueError, match="`name`"):
        node.name = ""


def test_assigning_bad_parents_raises() -> None:
    node = ChanceNode(name="x")
    with pytest.raises(ValueError, match="cannot list itself"):
        node.parents = ("x",)


def test_assigning_non_callable_dist_raises() -> None:
    node = ChanceNode(name="x")
    with pytest.raises(TypeError, match="must be callable"):
        node.dist = 42  # ty: ignore[invalid-assignment]


def test_assigning_none_dist_is_allowed() -> None:
    """Setting dist back to None returns the node to UNCONFIGURED."""
    node = ChanceNode(name="x", dist=_dist_matching)
    assert node.consistency is Consistency.CONSISTENT

    node.dist = None
    assert node.consistency is Consistency.UNCONFIGURED


# --- mutation helpers -------------------------------------------------------


def test_add_parent_appends() -> None:
    node = ChanceNode(name="x")
    node.add_parent("a")
    node.add_parent("b")

    assert node.parents == ("a", "b")


def test_add_parent_is_idempotent() -> None:
    node = ChanceNode(name="x")
    node.add_parent("a")
    node.add_parent("a")

    assert node.parents == ("a",)


def test_remove_parent_drops_match() -> None:
    node = ChanceNode(name="x", parents=("a", "b", "c"))
    node.remove_parent("b")

    assert node.parents == ("a", "c")


def test_remove_parent_absent_is_noop() -> None:
    node = ChanceNode(name="x", parents=("a",))
    node.remove_parent("z")

    assert node.parents == ("a",)


# --- consistency lifecycle --------------------------------------------------


def test_fresh_node_is_unconfigured() -> None:
    assert ChanceNode(name="x").consistency is Consistency.UNCONFIGURED


def test_matching_dist_is_consistent() -> None:
    node = ChanceNode(name="x", parents=("rain",), dist=_dist_taking_rain)
    assert node.consistency is Consistency.CONSISTENT


def test_kwargs_dist_matches_any_parents() -> None:
    """A **kwargs dist is the documented bypass for signature checking."""
    node = ChanceNode(name="x", parents=("a", "b", "c"), dist=_dist_matching)
    assert node.consistency is Consistency.CONSISTENT


def test_adding_parent_after_dist_makes_node_stale() -> None:
    node = ChanceNode(name="x", parents=("rain",), dist=_dist_taking_rain)
    assert node.consistency is Consistency.CONSISTENT

    node.add_parent("sprinkler")  # dist signature no longer matches
    assert node.consistency is Consistency.STALE


def test_fixing_dist_after_parent_change_restores_consistency() -> None:
    node = ChanceNode(name="x", parents=("rain",), dist=_dist_taking_rain)
    node.add_parent("sprinkler")
    assert node.consistency is Consistency.STALE

    node.dist = _dist_matching
    assert node.consistency is Consistency.CONSISTENT


def test_dist_with_extra_param_is_stale() -> None:
    """A dist expecting a parent that isn't in the parent set is stale."""

    def dist_taking_two(rain: Any, sprinkler: Any) -> object:  # noqa: ARG001
        return object()

    node = ChanceNode(name="x", parents=("rain",), dist=dist_taking_two)
    assert node.consistency is Consistency.STALE


# --- validate() gate --------------------------------------------------------


def test_validate_raises_on_unconfigured() -> None:
    node = ChanceNode(name="x")
    with pytest.raises(ValueError, match="no `dist`"):
        node.validate()


def test_validate_raises_on_stale() -> None:
    # _dist_taking_rain matches only (rain,); adding a parent makes it stale.
    node = ChanceNode(name="x", parents=("rain",), dist=_dist_taking_rain)
    node.add_parent("sprinkler")
    with pytest.raises(ValueError, match="does not match"):
        node.validate()


def test_validate_silent_on_consistent() -> None:
    node = ChanceNode(name="x", parents=("rain",), dist=_dist_taking_rain)
    node.validate()  # no exception


# --- is_discrete ------------------------------------------------------------


def test_is_discrete_follows_states() -> None:
    assert ChanceNode(name="x").is_discrete is False
    assert ChanceNode(name="x", states=("a",)).is_discrete is True
