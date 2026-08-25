"""Tests for :func:`decisionpy.inference.solve` — the intervention scan."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import InferenceError, solve

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


def _continuous_info_diagram() -> InfluenceDiagram:
    """A decision observing a continuous parent: an untabulated info set."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(DecisionNode(name="d", parents=("x",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u", parents=("x", "d"), values=lambda x, d: float(x) * float(d)
        )
    )
    return diag


def _mixed_diagram() -> InfluenceDiagram:
    """Discrete signal, continuous outcome: s -> x | s -> d(s) -> U."""
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


# --- solving -----------------------------------------------------------------


def test_solve_policy_is_optimal():
    """Umbrella should mirror rain: {(0,): 0, (1,): 1} with E[U] = 1.0."""
    solution = solve(_umbrella_diagram())
    assert solution.policy == {"umbrella": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(1.0, abs=0.05)


def test_solve_is_a_monte_carlo_estimate():
    """The MEU is a sample estimate; it wobbles with the seed."""
    import jax

    from decisionpy.inference.numpyro.solver import solve as numpyro_solve

    estimates = {
        numpyro_solve(
            _mixed_diagram().snapshot(), rng_key=jax.random.PRNGKey(seed)
        ).expected_utility
        for seed in range(3)
    }
    assert len(estimates) >= 2  # different seeds, different estimates
    assert all(value == pytest.approx(0.7, abs=0.15) for value in estimates)


def test_solve_no_decisions_raises():
    with pytest.raises(InferenceError, match="at least one decision"):
        solve(_bayesian_network())


def test_solve_mixed_diagram_solves():
    """
    Continuous chance nodes do not rule the solver out.

    E[x | s=0] = -1 → d=0; E[x | s=1] = 1 → d=1; E[U] = 0.7.
    """
    solution = solve(_mixed_diagram())
    assert solution.policy == {"d": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(0.7, abs=0.15)


def test_solve_continuous_information_set_raises():
    with pytest.raises(InferenceError, match="continuous"):
        solve(_continuous_info_diagram())
