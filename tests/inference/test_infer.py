"""Tests for :func:`pylimid.inference.infer`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.utility_node import UtilityNode
from pylimid.inference import InferenceError, Posterior, infer

# --- helpers -----------------------------------------------------------------


def _rain_wet_grass() -> InfluenceDiagram:
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
            dist=lambda rain: dist.Categorical(
                probs=jnp.array([[0.80, 0.20], [0.05, 0.95]])[rain]
            ),
        )
    )
    return diag


def _continuous_diagram() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=5.0, scale=2.0)))
    return diag


def _mixed_diagram() -> InfluenceDiagram:
    """Rain (discrete root) and temperature (continuous root)."""
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
            name="temp",
            dist=lambda: dist.Normal(loc=20.0, scale=3.0),
        )
    )
    return diag


def _decision_diagram() -> InfluenceDiagram:
    """A valid influence diagram with a decision and a utility node."""
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


def _umbrella_diagram() -> InfluenceDiagram:
    """``rain -> umbrella (decision) -> wet``; utility on ``(rain, umbrella)``."""
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
        ChanceNode(
            name="wet",
            parents=("rain", "umbrella"),
            states=("dry", "wet"),
            dist=lambda rain, umbrella: dist.Categorical(
                # Single indexing op: numpyro enumeration mis-weights
                # chained ``[rain][umbrella]`` indexing.
                probs=jnp.array(
                    [
                        [[0.90, 0.10], [0.70, 0.30]],  # rain=0
                        [[0.40, 0.60], [0.05, 0.95]],  # rain=1
                    ]
                )[rain, umbrella]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: float(rain == umbrella),
        )
    )
    return diag


# --- results -----------------------------------------------------------------


def test_discrete_prior():
    result = infer(_rain_wet_grass(), ["rain"])
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].is_discrete
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_discrete_posterior_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].marginal() == pytest.approx([0.457, 0.543], abs=0.04)


def test_result_keys_match_query():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_probabilities_sum_to_one():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert isinstance(result["rain"], Posterior)
    assert isinstance(result["wet_grass"], Posterior)
    assert sum(result["rain"].marginal()) == pytest.approx(1.0)
    assert sum(result["wet_grass"].marginal()) == pytest.approx(1.0)


def test_probabilities_sum_to_one_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert isinstance(result["rain"], Posterior)
    assert sum(result["rain"].marginal()) == pytest.approx(1.0)
    assert all(isinstance(p, float) for p in result["rain"].marginal())


def test_continuous_posterior_keeps_raw_draws():
    """Continuous posteriors store raw draws and carry no states."""
    result = infer(_continuous_diagram(), ["x"])
    assert isinstance(result["x"], Posterior)
    assert result["x"].states is None
    assert not result["x"].is_discrete
    assert len(result["x"].values) == 2000  # raw MCMC draws
    assert result["x"].mean() == pytest.approx(5.0, abs=0.2)
    assert result["x"].std() == pytest.approx(2.0, abs=0.15)
    lo, hi = result["x"].hdi()
    assert lo < 5.0 < hi


def test_mixed_result_shape_follows_variable_type():
    """One Posterior type; states distinguish discrete from continuous."""
    result = infer(_mixed_diagram(), ["rain", "temp"])
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)
    assert isinstance(result["temp"], Posterior)
    assert result["temp"].states is None
    assert len(result["temp"].values) == 2000  # raw MCMC draws
    assert result["temp"].mean() == pytest.approx(20.0, abs=0.3)


def test_discrete_posterior_stores_state_index_draws():
    """Discrete draws are integer state indices; the vector is derived."""
    result = infer(_rain_wet_grass(), ["rain"])
    assert all(v in (0.0, 1.0) for v in result["rain"].values)
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_summaries_on_discrete_raise():
    """Index draws are not moments: mean/std/hdi refuse discrete posteriors."""
    result = infer(_rain_wet_grass(), ["rain"])
    for method in ("mean", "std", "hdi"):
        with pytest.raises(ValueError, match="requires a continuous"):
            getattr(result["rain"], method)()


def test_marginal_on_continuous_raises():
    result = infer(_continuous_diagram(), ["x"])
    with pytest.raises(ValueError, match="marginal\\(\\) requires a discrete"):
        result["x"].marginal()


def test_hdi_invalid_coverage_raises():
    result = infer(_continuous_diagram(), ["x"])
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="coverage must be in"):
            result["x"].hdi(prob=bad)


# --- errors ------------------------------------------------------------------


def test_unbound_decision_raises():
    with pytest.raises(InferenceError, match="unbound decision"):
        infer(_decision_diagram(), ["rain"])


# --- policy binding -----------------------------------------------------------


def test_policy_collapses_id_to_bn():
    """
    A fully bound ID behaves like a BN with the decisions as evidence.

    Exact posterior: P(rain=yes | umbrella=1, wet=1) = 0.2 * 0.95 / 0.43
    ≈ 0.442.
    """
    result = infer(
        _umbrella_diagram(),
        ["rain"],
        observed={"wet": 1},
        policy={"umbrella": 1},
    )
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].marginal() == pytest.approx([0.558, 0.442], abs=0.04)


def test_policy_on_continuous_diagram():
    """A continuous chance node plus a bound decision still infers."""
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d", states=("no", "yes")))
    diag.add_node(
        ChanceNode(
            name="x", parents=("d",), dist=lambda d: dist.Normal(loc=2.0 * d, scale=1.0)
        )
    )
    diag.add_node(
        UtilityNode(
            name="u", parents=("x", "d"), values=lambda x, d: float(x) * float(d)
        )
    )
    result = infer(diag, ["x"], policy={"d": 1})
    assert isinstance(result["x"], Posterior)
    assert result["x"].states is None
    assert len(result["x"].values) == 2000
    assert result["x"].mean() == pytest.approx(2.0, abs=0.2)


def test_utility_node_alone_does_not_block_infer():
    """Utilities are ignored; a chance+utility diagram infers fine."""
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
            name="dryness",
            parents=("rain",),
            values=lambda rain: float(rain),
        )
    )
    result = infer(diag, ["rain"])
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_partially_bound_decisions_raise():
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d1", states=("no", "yes")))
    diag.add_node(DecisionNode(name="d2", states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u", parents=("d1", "d2"), values=lambda d1, d2: float(d1 + d2)
        )
    )
    with pytest.raises(InferenceError, match="unbound decision.*d2"):
        infer(diag, ["u"], policy={"d1": 0})


def test_policy_unknown_name_raises():
    with pytest.raises(InferenceError, match="unknown names.*bogus"):
        infer(_umbrella_diagram(), ["rain"], policy={"bogus": 0})


def test_policy_on_bayesian_network_raises():
    with pytest.raises(InferenceError, match="decision nodes only"):
        infer(_rain_wet_grass(), ["rain"], policy={"rain": 0})


def test_querying_a_decision_raises():
    with pytest.raises(InferenceError, match="chance variables only"):
        infer(_umbrella_diagram(), ["umbrella"], policy={"umbrella": 1})


def test_querying_a_utility_raises():
    with pytest.raises(InferenceError, match="chance variables only"):
        infer(_umbrella_diagram(), ["dryness"], policy={"umbrella": 1})
