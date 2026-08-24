"""Tests for :meth:`InfluenceDiagram.probe` — the parent-sensitivity check."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, InfluenceDiagram, UtilityNode
from decisionpy.graph.diagram import ProblemKind

# --- helpers -----------------------------------------------------------------


def _group_mixture(loc_table, scale_table) -> InfluenceDiagram:
    """A categorical parent ``group`` feeding a continuous child ``y``."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="group",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="y",
            parents=("group",),
            dist=lambda group: dist.Normal(
                loc=loc_table[group], scale=scale_table[group]
            ),
        )
    )
    return diag


def _rain_wet_grass(probs_table) -> InfluenceDiagram:
    """A discrete parent ``rain`` feeding a discrete child ``wet_grass``."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="wet_grass",
            parents=("rain",),
            states=("dry", "wet"),
            dist=lambda rain: dist.Categorical(probs=probs_table[rain]),
        )
    )
    return diag


# --- findings ----------------------------------------------------------------


def test_probe_catches_clamped_continuous_child():
    """1-row loc and scale tables clamped by JAX indexing: y never sees ``group``."""
    diag = _group_mixture(loc_table=jnp.array([10.0]), scale_table=jnp.array([1.0]))
    problems = diag.probe()
    assert len(problems) == 1
    assert problems[0].kind is ProblemKind.DIST_IGNORES_PARENT
    assert problems[0].node == "y"
    assert "group" in problems[0].message


def test_probe_catches_clamped_discrete_child():
    """A 1-row probs table: wet_grass never sees ``rain``."""
    diag = _rain_wet_grass(probs_table=jnp.array([[0.90, 0.10]]))
    problems = diag.probe()
    assert len(problems) == 1
    assert problems[0].kind is ProblemKind.DIST_IGNORES_PARENT
    assert problems[0].node == "wet_grass"


def test_probe_clean_on_correct_tables():
    diag = _group_mixture(
        loc_table=jnp.array([10.0, 20.0]), scale_table=jnp.array([1.0, 2.0])
    )
    assert diag.probe() == []


def test_probe_clean_on_correct_discrete_tables():
    diag = _rain_wet_grass(probs_table=jnp.array([[0.90, 0.10], [0.10, 0.90]]))
    assert diag.probe() == []


def test_probe_flags_utility_ignoring_a_parent():
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        UtilityNode(
            name="payoff",
            parents=("rain",),
            values=lambda rain: 1.0,  # ignores rain
        )
    )
    problems = diag.probe()
    assert len(problems) == 1
    assert problems[0].node == "payoff"


def test_probe_flags_intentionally_independent_node():
    """An independent node looks like a wiring mistake — warning stands."""
    diag = _rain_wet_grass(probs_table=jnp.array([[0.90, 0.10], [0.90, 0.10]]))
    problems = diag.probe()
    assert len(problems) == 1
    assert problems[0].kind is ProblemKind.DIST_IGNORES_PARENT


def test_probe_skips_continuous_parents():
    """Continuous parents have no states to enumerate — out of scope."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(
        ChanceNode(
            name="y",
            parents=("x",),
            dist=lambda x: dist.Normal(loc=0.0, scale=1.0),  # ignores x
        )
    )
    assert diag.probe() == []


def test_probe_cannot_detect_swapped_rows():
    """Wrong-but-varying numbers (swapped rows) are beyond the check."""
    diag = _rain_wet_grass(
        probs_table=jnp.array([[0.10, 0.90], [0.90, 0.10]])  # swapped
    )
    assert diag.probe() == []
