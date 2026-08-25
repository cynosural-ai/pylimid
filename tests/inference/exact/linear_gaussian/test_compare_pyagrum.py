"""
Validate decisionpy linear-Gaussian VE against pyAgrum exact inference.

Each test builds the same network in both libraries (pyAgrum via
pyagrum.clg.CLG + CLGVariableElimination) and checks that means and
variances agree to tight tolerance. The engine's arithmetic is float32,
so agreement is asserted at 1e-4.

Expected runtime: ~2-3 seconds (pyAgrum import is slow).
"""

from __future__ import annotations

import pytest

from decisionpy.inference.exact.linear_gaussian import query

from ._fixtures import Chain, LongChain, SingleRoot, VStructure

_TOL = 1e-4


def _check(fixture, query_vars, observed=None):
    snap = fixture.decisionpy().snapshot()
    result = query(snap, variables=query_vars, observed=observed or {})
    pg_result = fixture.pyagrum_query(query_vars, observed=observed)
    for var in query_vars:
        mean, variance = result[var]
        exp_mean, exp_var = pg_result[var]
        assert mean == pytest.approx(exp_mean, abs=_TOL)
        assert variance == pytest.approx(exp_var, abs=_TOL)


# -- single root --------------------------------------------------------------


def test_single_root_prior():
    _check(SingleRoot, ["x"])


# -- chain --------------------------------------------------------------------


def test_chain_prior():
    _check(Chain, ["x1", "x2", "x3"])


def test_chain_posterior_root_given_leaf():
    _check(Chain, ["x1"], {"x3": 0.4})


def test_chain_posterior_mid_given_leaf():
    _check(Chain, ["x2"], {"x3": -1.7})


# -- v-structure --------------------------------------------------------------


def test_vstruct_prior():
    _check(VStructure, ["x1", "x2", "y"])


def test_vstruct_condition_on_child():
    _check(VStructure, ["x1"], {"y": 0.1})


def test_vstruct_two_observations():
    _check(VStructure, ["y"], {"x1": 0.3, "x2": -0.8})


# -- long chain ---------------------------------------------------------------


def test_long_chain_prior():
    _check(LongChain, ["a", "b", "c", "d"])


def test_long_chain_deep_evidence():
    _check(LongChain, ["b"], {"d": 2.0, "a": -1.0})
