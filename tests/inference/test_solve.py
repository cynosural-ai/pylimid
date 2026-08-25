"""Tests for :func:`decisionpy.inference.solve` — engine-level solve dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import InferenceError, solve
from decisionpy.inference.exact.categorical import solve as id_solve

# --- helpers -----------------------------------------------------------------


def _umbrella_diagram() -> InfluenceDiagram:
    """``rain -> umbrella (decision)``; utility on ``(rain, umbrella)``."""
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


def _mixed_decision_diagram() -> InfluenceDiagram:
    """A continuous chance node rules bucket elimination out."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(DecisionNode(name="d", parents=("x",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u", parents=("x", "d"), values=lambda x, d: float(x) * float(d)
        )
    )
    return diag


def _bayesian_network() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    return diag


# --- dispatch ----------------------------------------------------------------


def test_solve_matches_bucket_elim():
    diag = _umbrella_diagram()
    assert solve(diag) == id_solve(diag.snapshot())


def test_solve_policy_is_optimal():
    """Umbrella should mirror rain: {(0,): 0, (1,): 1} with E[U] = 1.0."""
    solution = solve(_umbrella_diagram())
    assert solution.policy == {"umbrella": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(1.0)


def test_solve_no_decisions_raises():
    with pytest.raises(InferenceError, match="at least one decision"):
        solve(_bayesian_network())


def test_solve_mixed_diagram_raises():
    with pytest.raises(InferenceError, match="not implemented"):
        solve(_mixed_decision_diagram())


def test_solve_explicit_numpyro_raises_not_implemented():
    with pytest.raises(InferenceError, match="not implemented"):
        solve(_umbrella_diagram(), engine="numpyro")


def test_solve_explicit_bucket_elim_on_mixed_raises():
    with pytest.raises(InferenceError, match="all-categorical"):
        solve(_mixed_decision_diagram(), engine="bucket_elim")


def test_solve_unknown_engine_raises():
    with pytest.raises(InferenceError, match="Unknown engine"):
        solve(_umbrella_diagram(), engine="bogus")
