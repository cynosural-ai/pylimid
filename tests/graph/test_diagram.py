"""Tests for :mod:`pylimid.graph.diagram`."""

from typing import Any

import pytest

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot, InfluenceDiagram
from pylimid.graph.node import Consistency
from pylimid.graph.utility_node import UtilityNode
from pylimid.graph.validation import DiagramProblemKind

# --- helpers ----------------------------------------------------------------


def _dist_matching(**_: Any) -> object:
    """A dist factory for parentless chance nodes (accepts **kwargs)."""
    return object()


def _root() -> ChanceNode:
    return ChanceNode(name="rain", dist=_dist_matching, states=("no", "yes"))


def _child(dist_matching: bool = True) -> ChanceNode:
    """A child over `rain`; its dist names ``rain`` if ``dist_matching``."""
    if dist_matching:

        def dist(rain: Any) -> object:  # noqa: ARG001
            return object()

    else:

        def dist(other: Any) -> object:  # noqa: ARG001
            return object()

    return ChanceNode(
        name="wet_grass", parents=("rain",), dist=dist, states=("dry", "wet")
    )


def _consistent_rain_wetgrass() -> InfluenceDiagram:
    """A valid two-node BN: rain -> wet_grass, both configured."""
    diag = InfluenceDiagram()
    diag.add_node(_root())
    diag.add_node(_child())
    return diag


# --- registration & lookups -------------------------------------------------


def test_add_node_and_lookup() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))

    assert "rain" in diag
    assert len(diag) == 1
    assert diag["rain"].name == "rain"
    assert diag.names == ("rain",)


def test_duplicate_name_rejected() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))

    with pytest.raises(ValueError, match="already in the diagram"):
        diag.add_node(ChanceNode(name="rain", dist=_dist_matching))


def test_unknown_name_lookup_raises() -> None:
    diag = InfluenceDiagram()

    with pytest.raises(KeyError):
        _ = diag["ghost"]


def test_add_node_tolerates_dangling_parent() -> None:
    """A node may be added before its parents exist — caught at validate."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="wet", parents=("rain",), dist=_dist_matching))

    assert "wet" in diag
    assert "rain" not in diag


def test_iteration_yields_insertion_order() -> None:
    diag = _consistent_rain_wetgrass()
    assert list(diag) == ["rain", "wet_grass"]


# --- remove_node ------------------------------------------------------------


def test_remove_node_returns_it_and_drops_from_registry() -> None:
    diag = _consistent_rain_wetgrass()
    removed = diag.remove_node("rain")

    assert removed.name == "rain"
    assert "rain" not in diag
    assert len(diag) == 1


def test_remove_node_scrubs_survivor_parents() -> None:
    """Removing a referenced parent cleans up dangling edges."""
    diag = _consistent_rain_wetgrass()
    diag.remove_node("rain")

    assert diag["wet_grass"].parents == ()


def test_remove_unknown_node_raises() -> None:
    diag = InfluenceDiagram()
    with pytest.raises(KeyError):
        diag.remove_node("ghost")


# --- edges ------------------------------------------------------------------


def test_add_edge_links_parent_to_child() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))
    diag.add_node(ChanceNode(name="wet", dist=_dist_matching))

    diag.add_edge("rain", "wet")

    assert diag.parents_of("wet") == ("rain",)
    assert diag.children_of("rain") == ("wet",)


def test_add_edge_idempotent() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(ChanceNode(name="b", dist=_dist_matching))

    diag.add_edge("a", "b")
    diag.add_edge("a", "b")  # no-op

    assert diag.parents_of("b") == ("a",)


def test_add_edge_unknown_child_raises() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))

    with pytest.raises(KeyError):
        diag.add_edge("a", "ghost")


def test_add_edge_tolerates_dangling_parent() -> None:
    """Like a parent declared in add_node, a missing parent waits for validate."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))

    diag.add_edge("ghost", "a")

    assert diag.parents_of("a") == ("ghost",)
    kinds = {p.kind for p in diag.validate()}
    assert DiagramProblemKind.DANGLING_PARENT in kinds


def test_add_edge_self_loop_rejected() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))

    with pytest.raises(ValueError, match="itself"):
        diag.add_edge("a", "a")


def test_add_edge_accepts_cycle_and_validate_reports_it() -> None:
    """A -> b exists; b -> a is accepted, and validate names the cycle."""
    diag = InfluenceDiagram()
    for n in ("a", "b"):
        diag.add_node(ChanceNode(name=n, dist=_dist_matching))
    diag.add_edge("a", "b")

    diag.add_edge("b", "a")

    cycles = [p for p in diag.validate() if p.kind is DiagramProblemKind.CYCLE]
    assert len(cycles) == 1
    assert cycles[0].node in {"a", "b"}


def test_validate_names_a_longer_cycle_in_edge_order() -> None:
    """A -> b -> c -> a, plus d downstream of the cycle but not on it."""
    diag = InfluenceDiagram()
    for n in ("a", "b", "c", "d"):
        diag.add_node(ChanceNode(name=n, dist=_dist_matching))
    diag.add_edge("a", "b")
    diag.add_edge("b", "c")
    diag.add_edge("c", "a")
    diag.add_edge("c", "d")

    cycles = [p for p in diag.validate() if p.kind is DiagramProblemKind.CYCLE]
    assert len(cycles) == 1
    assert cycles[0].message == "Diagram contains a cycle: 'b' -> 'c' -> 'a' -> 'b'."


def test_reversing_an_arc_in_either_order() -> None:
    """Adding the new arc before removing the old one is a valid edit."""
    diag = InfluenceDiagram()
    for n in ("a", "b"):
        diag.add_node(ChanceNode(name=n, dist=_dist_matching))
    diag.add_edge("a", "b")

    diag.add_edge("b", "a")
    diag.remove_edge("a", "b")

    assert diag.parents_of("a") == ("b",)
    assert diag.topological_sort() == ("b", "a")


def test_remove_edge() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(ChanceNode(name="b", dist=_dist_matching))
    diag.add_edge("a", "b")

    diag.remove_edge("a", "b")

    assert diag.parents_of("b") == ()


def test_remove_edge_absent_is_noop() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(ChanceNode(name="b", dist=_dist_matching))

    diag.remove_edge("a", "b")  # no edge to remove; no error


# --- topological sort -------------------------------------------------------


def test_topological_sort_puts_parents_before_children() -> None:
    diag = InfluenceDiagram()
    for n in ("cloudy", "rain", "sprinkler", "wet_grass"):
        diag.add_node(ChanceNode(name=n, dist=_dist_matching))
    diag.add_edge("cloudy", "rain")
    diag.add_edge("cloudy", "sprinkler")
    diag.add_edge("rain", "wet_grass")
    diag.add_edge("sprinkler", "wet_grass")

    order = diag.topological_sort()
    pos = {name: i for i, name in enumerate(order)}

    for name in order:
        for parent in diag.parents_of(name):
            assert pos[parent] < pos[name]


# --- set_dist ---------------------------------------------------------------


def test_set_dist_updates_node() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain"))  # unconfigured
    assert diag["rain"].consistency is Consistency.UNCONFIGURED

    diag.set_dist("rain", _dist_matching)

    assert diag["rain"].consistency is Consistency.CONSISTENT


def test_set_dist_none_returns_to_unconfigured() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))

    diag.set_dist("rain", None)

    assert diag["rain"].consistency is Consistency.UNCONFIGURED


# --- validate() -------------------------------------------------------------


def test_validate_clean_diagram_returns_empty() -> None:
    diag = _consistent_rain_wetgrass()
    assert diag.validate() == []


def test_validate_reports_dangling_parent() -> None:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="wet",
            parents=("rain",),
            dist=lambda rain: object(),  # noqa: ARG005
        )
    )

    problems = diag.validate()
    assert len(problems) == 1
    assert problems[0].kind is DiagramProblemKind.DANGLING_PARENT
    assert problems[0].node == "wet"


def test_validate_reports_unconfigured_node() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain"))  # no dist

    problems = diag.validate()
    assert len(problems) == 1
    assert problems[0].kind is DiagramProblemKind.UNCONFIGURED
    assert problems[0].node == "rain"


def test_validate_reports_stale_node() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))
    diag.add_node(_child(dist_matching=False))  # stale dist

    problems = diag.validate()
    kinds = {p.kind for p in problems}
    assert DiagramProblemKind.STALE in kinds
    stale_nodes = [p.node for p in problems if p.kind is DiagramProblemKind.STALE]
    assert "wet_grass" in stale_nodes


def test_validate_reports_all_problems_at_once() -> None:
    """Two distinct issues surface together, not just the first."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching))
    diag.add_node(
        ChanceNode(
            name="wet",
            parents=("ghost",),  # dangling parent
        )
    )  # also unconfigured (no dist)

    problems = diag.validate()
    kinds = {p.kind for p in problems}
    assert DiagramProblemKind.DANGLING_PARENT in kinds
    assert DiagramProblemKind.UNCONFIGURED in kinds


def test_validate_detects_cycle_from_direct_node_mutation() -> None:
    """A node mutated via its own add_parent bypasses the diagram; validate sees it."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(ChanceNode(name="b", dist=_dist_matching))
    diag.add_edge("a", "b")
    # Sneak in the back-edge directly on the node, bypassing the diagram.
    diag["a"].add_parent("b")  # now a -> b -> a cycle

    problems = diag.validate()
    assert any(p.kind is DiagramProblemKind.CYCLE for p in problems)


# --- snapshot() -------------------------------------------------------------


def test_snapshot_returns_validated_view_in_topological_order() -> None:
    diag = _consistent_rain_wetgrass()
    snap = diag.snapshot()

    assert isinstance(snap, DiagramSnapshot)
    assert snap.order == ("rain", "wet_grass")
    assert [name for name, _ in snap.nodes] == ["rain", "wet_grass"]


def test_snapshot_raises_on_invalid_diagram() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain"))  # unconfigured

    with pytest.raises(ValueError, match="not valid"):
        diag.snapshot()


# --- adjacency is derived ---------------------------------------------------


def test_children_of_reflects_edge_changes() -> None:
    """children_of is derived on demand, so it tracks live mutations."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(ChanceNode(name="b", dist=_dist_matching))

    assert diag.children_of("a") == ()
    diag.add_edge("a", "b")
    assert diag.children_of("a") == ("b",)
    diag.remove_edge("a", "b")
    assert diag.children_of("a") == ()


# --- mixed-type diagrams (chance + decision + utility) ----------------------


def _utility_values(**_: Any) -> float:
    return 1.0


def _mixed_diagram() -> InfluenceDiagram:
    """
    A valid 4-node ID: rain -> invest (decision) -> payoff (utility).

    The decision observes ``rain`` (information set); the utility depends on
    both ``rain`` and the decision.
    """
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching, states=("no", "yes")))
    diag.add_node(
        DecisionNode(name="invest", parents=("rain",), states=("buy", "sell"))
    )
    diag.add_node(
        UtilityNode(
            name="payoff",
            parents=("rain", "invest"),
            values=lambda rain, invest: 1.0,  # noqa: ARG005
        )
    )
    return diag


def test_mixed_diagram_registers_all_three_node_types() -> None:
    diag = _mixed_diagram()

    assert {diag[n].kind.value for n in diag.names} == {"chance", "decision", "utility"}
    assert len(diag) == 3


def test_mixed_diagram_topological_order_respects_dependencies() -> None:
    """Rain (chance) before invest (decision observes rain) before payoff."""
    diag = _mixed_diagram()
    order = diag.topological_sort()
    pos = {name: i for i, name in enumerate(order)}

    assert pos["rain"] < pos["invest"] < pos["payoff"]


def test_mixed_diagram_snapshot_is_valid() -> None:
    diag = _mixed_diagram()
    snap = diag.snapshot()

    assert snap.order == ("rain", "invest", "payoff")
    # snapshot nodes carry the concrete node types, not a narrowed base.
    assert snap.nodes[1][0] == "invest"
    assert snap.nodes[2][1].is_sink is True


def test_add_edge_from_utility_reported_by_validate() -> None:
    """A utility node is a sink: an edge out of one is reported, not rejected."""
    diag = InfluenceDiagram()
    diag.add_node(UtilityNode(name="payoff", values=_utility_values))
    diag.add_node(ChanceNode(name="x", dist=_dist_matching))

    diag.add_edge("payoff", "x")

    sink_problems = [
        p for p in diag.validate() if p.kind is DiagramProblemKind.UTILITY_NOT_SINK
    ]
    assert [p.node for p in sink_problems] == ["payoff"]


def test_validate_reports_unconfigured_decision_and_utility() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching, states=("no", "yes")))
    diag.add_node(DecisionNode(name="invest", parents=("rain",)))  # no states
    diag.add_node(UtilityNode(name="payoff", parents=("rain",)))  # no values

    problems = diag.validate()
    kinds = {p.kind for p in problems}
    assert DiagramProblemKind.UNCONFIGURED in kinds
    unconfigured = {
        p.node for p in problems if p.kind is DiagramProblemKind.UNCONFIGURED
    }
    assert unconfigured == {"invest", "payoff"}


def test_validate_reports_utility_not_sink_declared_in_add_node() -> None:
    """A node that lists a utility among its parents is reported by validate."""
    diag = InfluenceDiagram()
    diag.add_node(UtilityNode(name="payoff", values=_utility_values))
    diag.add_node(
        ChanceNode(
            name="x", parents=("payoff",), dist=_dist_matching, states=("a", "b")
        )
    )

    problems = diag.validate()
    sink_problems = [
        p for p in problems if p.kind is DiagramProblemKind.UTILITY_NOT_SINK
    ]
    assert len(sink_problems) == 1
    assert sink_problems[0].node == "payoff"


def test_validate_detects_cycle_through_decision() -> None:
    """Decisions participate in cycle detection like any node."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", dist=_dist_matching))
    diag.add_node(DecisionNode(name="d", states=("x", "y")))
    diag.add_edge("d", "a")
    diag.add_edge("a", "d")  # closes a -> d -> a

    problems = diag.validate()
    assert any(p.kind is DiagramProblemKind.CYCLE for p in problems)


# --- set_dist narrowing -----------------------------------------------------


def test_set_dist_rejects_non_chance_node() -> None:
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d", states=("buy", "sell")))
    diag.add_node(UtilityNode(name="u", values=_utility_values))

    with pytest.raises(TypeError, match="chance nodes only"):
        diag.set_dist("d", _dist_matching)
    with pytest.raises(TypeError, match="chance nodes only"):
        diag.set_dist("u", _dist_matching)


# --- edges into utility nodes -----------------------------------------------


def test_add_edge_to_utility_as_child_is_allowed() -> None:
    """A utility gaining a *parent* is legitimate (it depends on that node)."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist_matching, states=("no", "yes")))
    diag.add_node(UtilityNode(name="payoff", values=_utility_values))

    diag.add_edge("rain", "payoff")  # no error
    assert diag.parents_of("payoff") == ("rain",)
