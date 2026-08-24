"""Tests for :mod:`decisionpy.graph.mermaid`."""

from typing import Any

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.utility_node import UtilityNode

# --- helpers ----------------------------------------------------------------


def _dist(**_: Any) -> object:
    """A dist factory accepting **kwargs (rendering never validates)."""
    return object()


def _values(**_: Any) -> float:
    """A utility values function accepting **kwargs."""
    return 0.0


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
    out = _bn().to_mermaid()

    assert '    rain(("rain"))' in out
    assert '    wet_grass(("wet_grass"))' in out


def test_decision_node_renders_as_rectangle() -> None:
    out = _influence().to_mermaid()

    assert '    invest["invest"]' in out


def test_utility_node_renders_as_diamond() -> None:
    out = _influence().to_mermaid()

    assert '    profit{"profit"}' in out


# --- edges & layout ---------------------------------------------------------


def test_edges_and_flow_direction() -> None:
    out = _influence().to_mermaid()

    assert out.startswith("flowchart LR\n")
    assert "    market --> invest" in out
    assert "    invest --> profit" in out


def test_topological_order_parents_before_children() -> None:
    out = _bn().to_mermaid()

    assert out.index('rain(("rain"))') < out.index('wet_grass(("wet_grass"))')


def test_output_is_deterministic() -> None:
    assert _influence().to_mermaid() == _influence().to_mermaid()


# --- incomplete workspace ---------------------------------------------------


def test_dangling_parent_renders_as_dashed_ghost() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="wet_grass", parents=("rain",), dist=_dist))

    out = diag.to_mermaid()

    assert '    rain["rain"]:::dangling' in out
    assert "    rain -.-> wet_grass" in out
    assert "    classDef dangling" in out


def test_dangling_parent_deduplicated_across_nodes() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a", parents=("x",), dist=_dist))
    diag.add_node(ChanceNode(name="b", parents=("x",), dist=_dist))

    out = diag.to_mermaid()

    assert out.count('x["x"]:::dangling') == 1


def test_unconfigured_and_stale_nodes_get_classes() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="unconfigured"))
    diag.add_node(ChanceNode(name="stale", dist=lambda other: object()))
    diag["stale"].add_parent("unconfigured")

    out = diag.to_mermaid()

    assert '    unconfigured(("unconfigured")):::unconfigured' in out
    assert '    stale(("stale")):::stale' in out
    assert "    class unconfigured unconfigured" in out
    assert "    class stale stale" in out
    assert "    classDef unconfigured" in out
    assert "    classDef stale" in out


def test_consistent_diagram_has_no_class_defs() -> None:
    out = _influence().to_mermaid()

    assert "classDef" not in out
    assert ":::" not in out


# --- name handling ----------------------------------------------------------


def test_quotes_in_names_are_escaped() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name='he said "hi"', dist=_dist))

    out = diag.to_mermaid()

    assert 'he_said__hi_(("he said &quot;hi&quot;"))' in out


def test_names_sanitizing_to_same_id_get_unique_ids() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="a b", dist=_dist))
    diag.add_node(ChanceNode(name="a-b", dist=_dist))

    out = diag.to_mermaid()

    assert out.count("a_b") >= 2
    assert "a_b_2" in out


def test_edges_use_sanitized_ids() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="some parent", dist=_dist))
    diag.add_node(ChanceNode(name="child", parents=("some parent",), dist=_dist))

    out = diag.to_mermaid()

    assert "    some_parent --> child" in out
