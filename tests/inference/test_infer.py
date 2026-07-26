"""Tests for :func:`decisionpy.inference.infer`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.inference import InferenceError, infer
from decisionpy.inference.ve import query as ve_query

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


# --- auto-dispatch -----------------------------------------------------------


def test_auto_discrete_dispatches_to_ve():
    result = infer(_rain_wet_grass(), ["rain"])
    expected = ve_query(_rain_wet_grass().snapshot(), variables=["rain"])
    assert result["rain"] == pytest.approx(expected["rain"])


def test_auto_with_observed():
    result = infer(
        _rain_wet_grass(),
        ["rain"],
        observed={"wet_grass": 1},
    )
    expected = ve_query(
        _rain_wet_grass().snapshot(),
        variables=["rain"],
        observed={"wet_grass": 1},
    )
    assert result["rain"] == pytest.approx(expected["rain"])


def test_auto_continuous_requires_numpyro():
    result = infer(_continuous_diagram(), ["x"], engine="numpyro")
    assert "x" in result


# --- explicit engine selection -----------------------------------------------


def test_explicit_ve():
    result = infer(_rain_wet_grass(), ["wet_grass"], engine="ve")
    expected = ve_query(_rain_wet_grass().snapshot(), variables=["wet_grass"])
    assert result["wet_grass"] == pytest.approx(expected["wet_grass"])


def test_explicit_numpyro_discrete():
    result = infer(
        _rain_wet_grass(),
        ["rain"],
        observed={"wet_grass": 1},
        engine="numpyro",
    )
    # P(rain=yes | wet_grass=wet) ≈ 0.543
    assert result["rain"] == pytest.approx([0.457, 0.543], abs=0.04)


def test_explicit_numpyro_prior():
    result = infer(_rain_wet_grass(), ["rain"], engine="numpyro")
    assert result["rain"] == pytest.approx([0.8, 0.2], abs=0.04)


# --- result format -----------------------------------------------------------


def test_result_keys_match_query():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_probabilities_sum_to_one():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert sum(result["rain"]) == pytest.approx(1.0)
    assert sum(result["wet_grass"]) == pytest.approx(1.0)


def test_probabilities_sum_to_one_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert sum(result["rain"]) == pytest.approx(1.0)


# --- errors ------------------------------------------------------------------


def test_unknown_engine_raises():
    with pytest.raises(InferenceError, match="Unknown engine"):
        infer(_rain_wet_grass(), ["rain"], engine="bogus")


def test_ve_on_continuous_raises():
    with pytest.raises(InferenceError, match="all-discrete"):
        infer(_continuous_diagram(), ["x"], engine="ve")
