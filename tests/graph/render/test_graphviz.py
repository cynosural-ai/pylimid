"""Tests for :mod:`pylimid.graph.render.graphviz`."""

from typing import Any

import pytest

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.render import graphviz as graphviz_render
from pylimid.graph.utility_node import UtilityNode

# --- helpers ----------------------------------------------------------------


def _dist(**_: Any) -> object:
    """A dist factory accepting **kwargs (rendering never validates)."""
    return object()


def _values(**_: Any) -> float:
    """A utility values function accepting **kwargs."""
    return 0.0


def _dot(diagram: InfluenceDiagram) -> str:
    """The Graphviz renderer's internal DOT source."""
    return graphviz_render._to_dot(diagram)


def _bn() -> InfluenceDiagram:
    """A configured two-node BN: rain -> wet_grass."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="rain", dist=_dist, states=("no", "yes")))
    diag.add_node(
        ChanceNode(
            name="wet_grass",
            parents=("rain",),
            dist=lambda rain: object(),  # noqa: ARG005
            states=("dry", "wet"),
        )
    )
    return diag


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


# --- shapes -----------------------------------------------------------------


def test_chance_node_renders_as_circle() -> None:
    out = _dot(_bn())

    assert '    rain [label="rain", shape=circle];' in out
    assert '    wet_grass [label="wet_grass", shape=circle];' in out


def test_decision_node_renders_as_box() -> None:
    out = _dot(_influence())

    assert '    invest [label="invest", shape=box];' in out


def test_utility_node_renders_as_diamond() -> None:
    out = _dot(_influence())

    assert '    profit [label="profit", shape=diamond];' in out


# --- edges & layout ---------------------------------------------------------


def test_edges_and_flow_direction() -> None:
    out = _dot(_influence())

    assert out.startswith("digraph {\n")
    assert "    rankdir=LR;" in out
    assert "    market -> invest;" in out
    assert "    invest -> profit;" in out


def test_topological_order_parents_before_children() -> None:
    out = _dot(_bn())

    assert out.index('rain [label="rain"') < out.index('wet_grass [label="wet_grass"')


def test_output_is_deterministic() -> None:
    assert _dot(_influence()) == _dot(_influence())


# --- incomplete workspace ---------------------------------------------------


def test_dangling_parent_renders_as_dashed_ghost() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="wet_grass", parents=("rain",), dist=_dist))

    out = _dot(diag)

    assert '    rain [label="rain", shape=box, style="filled,dashed"' in out
    assert "    rain -> wet_grass [style=dashed];" in out


def test_dangling_parent_deduplicated_across_nodes() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", parents=("x",), dist=_dist))
    diag.add_node(ChanceNode(name="b", parents=("x",), dist=_dist))

    out = _dot(diag)

    assert out.count('    x [label="x"') == 1


def test_unconfigured_and_stale_nodes_get_styles() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="unconfigured"))
    diag.add_node(ChanceNode(name="stale", dist=lambda other: object()))
    diag["stale"].add_parent("unconfigured")

    out = _dot(diag)

    assert '    unconfigured [label="unconfigured", shape=circle, style=filled' in out
    assert 'fillcolor="#e9ecef"' in out
    assert '    stale [label="stale", shape=circle, style=filled' in out
    assert 'fillcolor="#fff3cd"' in out


def test_consistent_diagram_has_no_style() -> None:
    out = _dot(_influence())

    assert "style=" not in out
    assert "fillcolor" not in out


# --- name handling ----------------------------------------------------------


def test_quotes_in_names_are_escaped() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name='he said "hi"', dist=_dist))

    out = _dot(diag)

    assert 'he_said__hi_ [label="he said \\"hi\\"", shape=circle];' in out


def test_names_sanitizing_to_same_id_get_unique_ids() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a b", dist=_dist))
    diag.add_node(ChanceNode(name="a-b", dist=_dist))

    out = _dot(diag)

    assert out.count("a_b") >= 2
    assert "a_b_2" in out


def test_edges_use_sanitized_ids() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="some parent", dist=_dist))
    diag.add_node(ChanceNode(name="child", parents=("some parent",), dist=_dist))

    out = _dot(diag)

    assert "    some_parent -> child;" in out


# --- SVG --------------------------------------------------------------------


@pytest.mark.skipif(not graphviz_render.available(), reason="Graphviz is not installed")
def test_to_svg_renders_svg() -> None:
    svg = _influence().to_svg()

    assert "<svg" in svg


@pytest.mark.skipif(not graphviz_render.available(), reason="Graphviz is not installed")
def test_repr_svg_uses_graphviz() -> None:
    assert _influence()._repr_svg_() is not None
