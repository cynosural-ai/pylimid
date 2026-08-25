"""Tests for :func:`decisionpy.inference.numpyro.solver.solve`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import InferenceError, solve
from decisionpy.inference.numpyro.solver import solve as numpyro_solve

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


# --- categorical: agrees with bucket elimination -----------------------------


def test_umbrella_policy_matches_bucket_elim():
    exact = solve(_umbrella_diagram())
    approx = solve(_umbrella_diagram(), engine="numpyro")
    assert approx.policy == exact.policy == {"umbrella": {(0,): 0, (1,): 1}}
    assert approx.expected_utility == pytest.approx(1.0, abs=0.05)
    assert exact.exact is True
    assert approx.exact is False


def test_two_decisions_match_bucket_elim():
    exact = solve(_two_decisions_diagram())
    approx = solve(_two_decisions_diagram(), engine="numpyro")
    assert approx.expected_utility == pytest.approx(exact.expected_utility, abs=0.05)
    assert approx.policy["d1"] == exact.policy["d1"] == {(0,): 0, (1,): 1}
    # d2's optimum is unique only on reachable cells: under d1 = x, the
    # assignments (x=0, d1=1) and (x=1, d1=0) never occur, so any action
    # there ties and the two solvers may break the tie differently.
    for assignment in ((0, 0), (1, 1)):
        assert approx.policy["d2"][assignment] == exact.policy["d2"][assignment]


# --- mixed diagrams ----------------------------------------------------------


def test_mixed_diagram_optimal_policy():
    solution = solve(_mixed_diagram(), engine="numpyro")
    # E[x | s=0] = -1 → d=0; E[x | s=1] = 1 → d=1; E[U] = 0.7.
    assert solution.policy == {"d": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(0.7, abs=0.15)
    assert solution.exact is False


def test_mixed_diagram_bucket_elim_cannot_handle():
    with pytest.raises(InferenceError, match="all-categorical"):
        solve(_mixed_diagram())


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
    with pytest.raises(InferenceError, match="continuous"):
        solve(diag, engine="numpyro")


def test_solver_is_deterministic():
    a = numpyro_solve(_mixed_diagram().snapshot())
    b = numpyro_solve(_mixed_diagram().snapshot())
    assert a == b
