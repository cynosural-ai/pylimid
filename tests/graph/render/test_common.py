"""Tests for :mod:`pylimid.graph.render._common`."""

from typing import Any

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.render._common import Shape, Style, build_scene
from pylimid.graph.utility_node import UtilityNode

# --- helpers ----------------------------------------------------------------


def _dist(**_: Any) -> object:
    """A dist factory accepting **kwargs (rendering never validates)."""
    return object()


def _influence() -> InfluenceDiagram:
    """A configured influence diagram: chance -> decision -> utility."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="market", dist=_dist, states=("down", "up")))
    diag.add_node(
        DecisionNode(name="invest", parents=("market",), states=("no", "yes"))
    )
    diag.add_node(
        UtilityNode(
            name="profit",
            parents=("invest",),
            values=lambda invest: 0.0,  # noqa: ARG005
        )
    )
    return diag


# --- scene ------------------------------------------------------------------


def test_shapes_follow_node_kind() -> None:
    scene = build_scene(_influence())
    by_name = {node.name: node for node in scene.nodes}

    assert by_name["market"].shape is Shape.CIRCLE
    assert by_name["invest"].shape is Shape.RECTANGLE
    assert by_name["profit"].shape is Shape.DIAMOND


def test_consistent_nodes_have_no_style() -> None:
    scene = build_scene(_influence())

    assert all(node.style is None for node in scene.nodes)


def test_styles_follow_consistency() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="unconfigured"))
    diag.add_node(ChanceNode(name="stale", dist=lambda other: object()))
    diag["stale"].add_parent("unconfigured")

    scene = build_scene(diag)
    by_name = {node.name: node for node in scene.nodes}

    assert by_name["unconfigured"].style is Style.UNCONFIGURED
    assert by_name["stale"].style is Style.STALE


def test_dangling_parent_becomes_ghost_node_and_dashed_edge() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="wet_grass", parents=("rain",), dist=_dist))

    scene = build_scene(diag)
    by_name = {node.name: node for node in scene.nodes}

    assert by_name["rain"].shape is Shape.RECTANGLE
    assert by_name["rain"].style is Style.DANGLING
    assert [(edge.parent, edge.dangling) for edge in scene.edges] == [
        (by_name["rain"].id, True)
    ]


def test_nodes_are_in_topological_order() -> None:
    scene = build_scene(_influence())

    assert [node.name for node in scene.nodes] == ["market", "invest", "profit"]
