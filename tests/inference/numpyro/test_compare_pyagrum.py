"""
Validate the numpyro infer() against pyAgrum's exact LazyPropagation.

Each test builds the same categorical Bayesian network in both libraries
and checks that the bincounted posterior estimates agree with pyAgrum's
exact marginals. The engine's estimates are Monte-Carlo (2000 enumerated
draws), so agreement is asserted at abs=0.04 rather than 1e-6.

Expected runtime: ~2-3 seconds (pyAgrum import is slow).
"""

from __future__ import annotations

import pytest

from decisionpy.inference import infer

from ._bn_fixtures import FourNodeAsia, ThreeNodeChain, TwoNodeRainWet, VStructure

_TOL = 0.04


def _check(fixture, query_vars, observed=None):
    result = infer(fixture.decisionpy(), query_vars, observed=observed or {})
    expected = fixture.pyagrum_query(query_vars, observed=observed)
    for var in query_vars:
        assert result[var].marginal() == pytest.approx(expected[var], abs=_TOL), (
            f"Mismatch on {var!r} for {fixture.__class__.__name__}\n"
            f"  observed={observed}\n"
            f"  numpyro:  {result[var].marginal()}\n"
            f"  pyagrum:  {expected[var]}"
        )


# -- two-node BN: rain -> wet_grass -------------------------------------------


def test_two_node_prior():
    _check(TwoNodeRainWet, ["rain", "wet_grass"])


def test_two_node_posterior_rain_given_wet():
    _check(TwoNodeRainWet, ["rain"], {"wet_grass": 1})


def test_two_node_posterior_wet_given_rain():
    _check(TwoNodeRainWet, ["wet_grass"], {"rain": 1})


# -- three-node chain: a -> b -> c --------------------------------------------


def test_chain_prior_c():
    _check(ThreeNodeChain, ["c"])


def test_chain_posterior_a_given_c():
    _check(ThreeNodeChain, ["a"], {"c": 1})


# -- v-structure: a -> c <- b -------------------------------------------------


def test_vstruct_prior():
    _check(VStructure, ["c"])


def test_vstruct_explaining_away():
    _check(VStructure, ["a"], {"b": 1, "c": 0})


# -- four-node asia toy -------------------------------------------------------


def test_asia_prior_all():
    _check(FourNodeAsia, ["smoking", "pollution", "cancer", "xray"])


def test_asia_posterior_smoking_given_cancer_and_xray():
    _check(FourNodeAsia, ["smoking"], {"cancer": 1, "xray": 1})
