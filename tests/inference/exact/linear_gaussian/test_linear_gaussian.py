"""Tests for :func:`decisionpy.inference.exact.linear_gaussian.query`."""

from __future__ import annotations

import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.inference.exact.linear_gaussian import query

from ._fixtures import Chain, LongChain, SingleRoot, VStructure

#: The engine's arithmetic is float32 (JAX default), so agreement with the
#: float64 oracle is asserted at 1e-4 — a genuine algebra bug deviates by
#: orders of magnitude more.
_TOL = 1e-4


def _check(fixture, query_vars, observed=None):
    snap = fixture.decisionpy().snapshot()
    result = query(snap, variables=query_vars, observed=observed or {})
    expected = fixture.oracle_query(query_vars, observed=observed)
    for var in query_vars:
        mean, variance = result[var]
        exp_mean, exp_var = expected[var]
        assert mean == pytest.approx(exp_mean, abs=_TOL)
        assert variance == pytest.approx(exp_var, abs=_TOL)


# -- single root --------------------------------------------------------------


def test_single_root_prior():
    _check(SingleRoot, ["x"])


def test_single_root_observed_is_point_mass():
    _check(SingleRoot, ["x"], {"x": 1.5})


# -- chain --------------------------------------------------------------------


def test_chain_prior_all():
    _check(Chain, ["x1", "x2", "x3"])


def test_chain_prior_one():
    _check(Chain, ["x3"])


def test_chain_posterior_mid_given_leaf():
    _check(Chain, ["x2"], {"x3": 0.4})


def test_chain_posterior_root_given_mid():
    _check(Chain, ["x1"], {"x2": -0.7})


def test_chain_posterior_all_given_leaf():
    _check(Chain, ["x1", "x2"], {"x3": 1.2})


def test_chain_observed_query_is_point_mass():
    _check(Chain, ["x3"], {"x3": -2.0})


# -- v-structure --------------------------------------------------------------


def test_vstruct_prior():
    _check(VStructure, ["x1", "x2", "y"])


def test_vstruct_posterior_with_two_observations():
    _check(VStructure, ["x1"], {"x2": 2.0, "y": 0.1})


def test_vstruct_condition_on_child_only():
    _check(VStructure, ["x1", "x2"], {"y": -0.5})


def test_vstruct_condition_on_root():
    _check(VStructure, ["y"], {"x1": 0.3})


# -- long chain ---------------------------------------------------------------


def test_long_chain_prior():
    _check(LongChain, ["a", "b", "c", "d"])


def test_long_chain_deep_evidence():
    _check(LongChain, ["a"], {"c": 0.5, "d": -1.0})


# -- loud failures ------------------------------------------------------------


def test_non_linear_cpd_raises():
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode("x", dist=lambda: dist.Normal(0.0, 1.0)))
    diag.add_node(
        ChanceNode(
            "y",
            parents=("x",),
            dist=lambda x: dist.Normal(x**2, 1.0),
        )
    )
    with pytest.raises(ValueError, match="not linear"):
        query(diag.snapshot(), variables=["x"])


def test_discrete_cpd_raises():
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode("x", states=("a", "b"), dist=lambda: dist.Categorical([0.5, 0.5]))
    )
    with pytest.raises(TypeError, match="not a Normal"):
        query(diag.snapshot(), variables=["x"])
