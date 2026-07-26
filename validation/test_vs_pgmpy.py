"""Validate decisionpy VE against pgmpy exact inference.

Each test builds the same Bayesian network in both libraries and checks
that priors and posteriors agree to high precision (absolute tolerance 1e-6).

Expected runtime: ~2-3 seconds (pgmpy import is slow).
"""

from __future__ import annotations

import pytest

from decisionpy.inference.variable_elim import query as dp_query
from validation._fixtures import (
    FourNodeAsia,
    ThreeNodeChain,
    TwoNodeRainWet,
    VStructure,
)

# -- shared fixture factory ---------------------------------------------------


def _check(
    fixture,
    query_vars: list[str],
    observed: dict[str, int] | None = None,
):
    """Assert dp_query ≈ pgmpy_query for the given fixture and evidence."""
    snap = fixture.decisionpy().snapshot()
    dp_result = dp_query(snap, variables=query_vars, observed=observed or {})
    pg_result = fixture.pgmpy_query(query_vars, observed=observed)
    for var in query_vars:
        assert dp_result[var] == pytest.approx(pg_result[var], abs=1e-6), (
            f"Mismatch on {var!r} for {fixture.__class__.__name__}\n"
            f"  observed={observed}\n"
            f"  decisionpy: {dp_result[var]}\n"
            f"  pgmpy:      {pg_result[var]}"
        )


# -- two-node BN: rain -> wet_grass -------------------------------------------


def test_two_node_prior_rain():
    _check(TwoNodeRainWet, ["rain"])


def test_two_node_prior_wet_grass():
    _check(TwoNodeRainWet, ["wet_grass"])


def test_two_node_joint():
    _check(TwoNodeRainWet, ["rain", "wet_grass"])


def test_two_node_posterior_rain_given_dry():
    _check(TwoNodeRainWet, ["rain"], {"wet_grass": 0})


def test_two_node_posterior_rain_given_wet():
    _check(TwoNodeRainWet, ["rain"], {"wet_grass": 1})


def test_two_node_posterior_wet_given_rain():
    _check(TwoNodeRainWet, ["wet_grass"], {"rain": 1})


# -- three-node chain: a -> b -> c --------------------------------------------


def test_chain_prior_c():
    _check(ThreeNodeChain, ["c"])


def test_chain_posterior_a_given_c():
    _check(ThreeNodeChain, ["a"], {"c": 1})


def test_chain_posterior_a_given_c_0():
    _check(ThreeNodeChain, ["a"], {"c": 0})


# -- v-structure: a -> c <- b -------------------------------------------------


def test_vstruct_prior():
    _check(VStructure, ["c"])


def test_vstruct_explaining_away():
    _check(VStructure, ["a"], {"b": 1, "c": 0})


def test_vstruct_both_observed():
    _check(VStructure, ["a", "b"], {"c": 1})


# -- four-node asia toy -------------------------------------------------------


def test_asia_prior_all():
    _check(FourNodeAsia, ["smoking", "pollution", "cancer", "xray"])


def test_asia_posterior_cancer_given_xray_pos():
    _check(FourNodeAsia, ["cancer"], {"xray": 1})


def test_asia_posterior_smoking_given_cancer_and_xray():
    _check(FourNodeAsia, ["smoking"], {"cancer": 1, "xray": 1})


def test_asia_posterior_pollution_given_smoking():
    _check(FourNodeAsia, ["pollution"], {"smoking": 1})


def test_asia_joint_smoking_pollution():
    _check(FourNodeAsia, ["smoking", "pollution"])
