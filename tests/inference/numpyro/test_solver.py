"""Tests for :func:`decisionpy.inference.numpyro.solver.solve`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference.numpyro.solver import solve

# --- helpers ----------------------------------------------------------------


def _umbrella_diagram() -> InfluenceDiagram:
    """``rain -> umbrella``; utility 1 iff the umbrella matches the rain."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        DecisionNode(name="umbrella", parents=("rain",), states=("no", "yes"))
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: float(rain == umbrella),
        )
    )
    return diag


def _two_decisions_diagram() -> InfluenceDiagram:
    """Nested decisions; ``U = [d1 == x] + [d2 == d1]`` — a unique optimum."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="x",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="d1", parents=("x",), states=("no", "yes")))
    diag.add_node(DecisionNode(name="d2", parents=("x", "d1"), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u",
            parents=("x", "d1", "d2"),
            values=lambda x, d1, d2: float(d1 == x) + float(d2 == d1),
        )
    )
    return diag


def _mixed_diagram() -> InfluenceDiagram:
    """Discrete signal s, continuous x | s, decision on s, utility x*d."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="s",
            states=("low", "high"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.3, 0.7])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="x",
            parents=("s",),
            dist=lambda s: dist.Normal(loc=-1.0 + 2.0 * s, scale=1.0),
        )
    )
    diag.add_node(DecisionNode(name="d", parents=("s",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u",
            parents=("x", "d"),
            values=lambda x, d: float(x) * float(d),
        )
    )
    return diag


# --- categorical -------------------------------------------------------------


def test_umbrella_policy():
    solution = solve(_umbrella_diagram().snapshot())
    assert solution.policy == {"umbrella": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(1.0, abs=0.05)


def test_two_decisions():
    solution = solve(_two_decisions_diagram().snapshot())
    assert solution.policy["d1"] == {(0,): 0, (1,): 1}
    assert solution.expected_utility == pytest.approx(2.0, abs=0.05)


# --- mixed diagrams ----------------------------------------------------------


def test_mixed_diagram_optimal_policy():
    solution = solve(_mixed_diagram().snapshot())
    # E[x | s=0] = -1 → d=0; E[x | s=1] = 1 → d=1; E[U] = 0.7.
    assert solution.policy == {"d": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(0.7, abs=0.15)


# --- scope and determinism ---------------------------------------------------


def test_continuous_information_set_raises():
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(DecisionNode(name="d", parents=("x",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u",
            parents=("x", "d"),
            values=lambda x, d: float(x) * float(d),
        )
    )
    with pytest.raises(ValueError, match="continuous"):
        solve(diag.snapshot())


def test_solver_is_deterministic():
    a = solve(_mixed_diagram().snapshot())
    b = solve(_mixed_diagram().snapshot())
    assert a == b
