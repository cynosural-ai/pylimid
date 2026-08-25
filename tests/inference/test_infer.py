"""Tests for :func:`decisionpy.inference.infer`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference import (
    Draws,
    Gaussian,
    InferenceError,
    Marginal,
    infer,
)
from decisionpy.inference.exact.categorical import query as ve_query

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


def _lg_diagram() -> InfluenceDiagram:
    """A two-node linear-Gaussian BN: x1 -> x2."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x1", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(
        ChanceNode(
            name="x2",
            parents=("x1",),
            dist=lambda x1: dist.Normal(loc=2.0 + 1.5 * x1, scale=0.5),
        )
    )
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


# --- default engine (numpyro) ------------------------------------------------


def test_default_engine_is_numpyro():
    """infer() defaults to the Monte-Carlo engine; results are never exact."""
    result = infer(_rain_wet_grass(), ["rain"])
    assert isinstance(result["rain"], Marginal)
    assert result["rain"].values == pytest.approx([0.8, 0.2], abs=0.04)
    assert result["rain"].exact is False


def test_default_engine_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert isinstance(result["rain"], Marginal)
    assert result["rain"].values == pytest.approx([0.457, 0.543], abs=0.04)
    assert result["rain"].exact is False


def test_default_engine_on_continuous_returns_draws():
    result = infer(_continuous_diagram(), ["x"])
    assert isinstance(result["x"], Draws)


# --- explicit engine selection -----------------------------------------------


def test_explicit_ve():
    result = infer(_rain_wet_grass(), ["wet_grass"], engine="ve")
    expected = ve_query(_rain_wet_grass().snapshot(), variables=["wet_grass"])
    assert result["wet_grass"] == Marginal(values=expected["wet_grass"], exact=True)


def test_explicit_numpyro_discrete():
    result = infer(
        _rain_wet_grass(),
        ["rain"],
        observed={"wet_grass": 1},
        engine="numpyro",
    )
    # P(rain=yes | wet_grass=wet) ≈ 0.543
    assert isinstance(result["rain"], Marginal)
    assert result["rain"].values == pytest.approx([0.457, 0.543], abs=0.04)
    assert result["rain"].exact is False


def test_explicit_numpyro_prior():
    result = infer(_rain_wet_grass(), ["rain"], engine="numpyro")
    assert isinstance(result["rain"], Marginal)
    assert result["rain"].values == pytest.approx([0.8, 0.2], abs=0.04)


# --- result format -----------------------------------------------------------


def test_result_keys_match_query():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"], engine="ve")
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_probabilities_sum_to_one():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"], engine="ve")
    assert isinstance(result["rain"], Marginal)
    assert isinstance(result["wet_grass"], Marginal)
    assert sum(result["rain"].values) == pytest.approx(1.0)
    assert sum(result["wet_grass"].values) == pytest.approx(1.0)


def test_probabilities_sum_to_one_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1}, engine="ve")
    assert isinstance(result["rain"], Marginal)
    assert sum(result["rain"].values) == pytest.approx(1.0)


def test_engine_choice_does_not_change_result_shape():
    """Discrete queries are Marginals no matter which engine runs."""
    ve_result = infer(_rain_wet_grass(), ["rain"], engine="ve")
    np_result = infer(_rain_wet_grass(), ["rain"], engine="numpyro")
    assert isinstance(ve_result["rain"], Marginal)
    assert isinstance(np_result["rain"], Marginal)
    assert len(ve_result["rain"].values) == len(np_result["rain"].values) == 2
    assert sum(np_result["rain"].values) == pytest.approx(1.0)
    assert all(isinstance(p, float) for p in np_result["rain"].values)


def test_engine_choice_changes_exactness():
    """The ``exact`` flag carries what the engine name would have told you."""
    ve_result = infer(_rain_wet_grass(), ["rain"], engine="ve")
    np_result = infer(_rain_wet_grass(), ["rain"], engine="numpyro")
    assert isinstance(ve_result["rain"], Marginal)
    assert isinstance(np_result["rain"], Marginal)
    assert ve_result["rain"].exact is True
    assert np_result["rain"].exact is False


def test_continuous_returns_draws():
    """Continuous queries return Draws, not probabilities."""
    result = infer(_continuous_diagram(), ["x"], engine="numpyro")
    assert isinstance(result["x"], Draws)
    assert len(result["x"].values) == 2000  # raw MCMC draws
    assert sum(result["x"].values) != pytest.approx(1.0)  # not probabilities


def test_mixed_result_shape_follows_variable_type():
    """Discrete queries return Marginals; continuous return Draws."""
    result = infer(_mixed_diagram(), ["rain", "temp"], engine="numpyro")
    assert isinstance(result["rain"], Marginal)
    assert sum(result["rain"].values) == pytest.approx(1.0)  # probability vector
    assert isinstance(result["temp"], Draws)
    assert len(result["temp"].values) == 2000  # raw MCMC draws
    assert sum(result["temp"].values) != pytest.approx(1.0)  # not probabilities


# --- linear-Gaussian engine --------------------------------------------------


def test_lg_returns_gaussian():
    result = infer(_lg_diagram(), ["x2"], engine="lg")
    assert isinstance(result["x2"], Gaussian)
    assert result["x2"].mean == pytest.approx(2.0)
    assert result["x2"].variance == pytest.approx(2.5)


def test_lg_posterior():
    result = infer(_lg_diagram(), ["x1"], observed={"x2": 0.4}, engine="lg")
    assert isinstance(result["x1"], Gaussian)
    assert result["x1"].mean == pytest.approx(-0.96, abs=1e-4)
    assert result["x1"].variance == pytest.approx(0.1, abs=1e-4)


def test_lg_observed_query_is_point_mass():
    result = infer(_lg_diagram(), ["x2"], observed={"x2": 1.5}, engine="lg")
    assert isinstance(result["x2"], Gaussian)
    assert result["x2"].mean == 1.5
    assert result["x2"].variance == 0.0


def test_lg_on_discrete_raises():
    with pytest.raises(InferenceError, match="all-continuous"):
        infer(_rain_wet_grass(), ["rain"], engine="lg")


def test_lg_on_non_linear_cpd_raises():
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0.0, 1.0)))
    diag.add_node(
        ChanceNode("y", parents=("x",), dist=lambda x: dist.Normal(x**2, 1.0))
    )
    with pytest.raises(InferenceError, match="linear-Gaussian"):
        infer(diag, ["x"], engine="lg")


# --- errors ------------------------------------------------------------------


def test_unknown_engine_raises():
    with pytest.raises(InferenceError, match="Unknown engine"):
        infer(_rain_wet_grass(), ["rain"], engine="bogus")


def test_ve_on_continuous_raises():
    with pytest.raises(InferenceError, match="all-discrete"):
        infer(_continuous_diagram(), ["x"], engine="ve")


def test_unbound_decision_raises():
    with pytest.raises(InferenceError, match="unbound decision"):
        infer(_decision_diagram(), ["rain"])


def test_unbound_decision_raises_with_explicit_engine():
    for engine in ("ve", "numpyro"):
        with pytest.raises(InferenceError, match="unbound decision"):
            infer(_decision_diagram(), ["rain"], engine=engine)


# --- policy binding -----------------------------------------------------------


def test_policy_collapses_id_to_bn():
    """A fully bound ID behaves like a BN with the decisions as evidence."""
    diag = _umbrella_diagram()
    result = infer(
        diag, ["rain"], observed={"wet": 1}, policy={"umbrella": 1}, engine="ve"
    )
    expected = ve_query(
        diag.snapshot(),
        variables=["rain"],
        observed={"umbrella": 1, "wet": 1},
    )
    assert result["rain"] == Marginal(values=expected["rain"], exact=True)


def test_policy_with_explicit_numpyro():
    """The numpyro engine sees the bound decision as an observed site."""
    diag = _umbrella_diagram()
    result = infer(
        diag,
        ["rain"],
        observed={"wet": 1},
        policy={"umbrella": 1},
        engine="numpyro",
    )
    expected = ve_query(
        diag.snapshot(),
        variables=["rain"],
        observed={"umbrella": 1, "wet": 1},
    )
    assert isinstance(result["rain"], Marginal)
    assert result["rain"].values == pytest.approx(expected["rain"], abs=0.04)
    assert result["rain"].exact is False


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
    assert isinstance(result["x"], Draws)
    assert len(result["x"].values) == 2000
    assert jnp.mean(jnp.array(result["x"].values)) == pytest.approx(2.0, abs=0.2)


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
    result = infer(diag, ["rain"], engine="ve")
    assert result["rain"] == Marginal(values=[0.8, 0.2], exact=True)


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
