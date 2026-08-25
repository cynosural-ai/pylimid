"""
Validate numpyro MCMC against pyAgrum's exact CLG inference (pyagrum.clg).

Each test builds the same all-continuous network in both libraries and
checks that the posterior draws' mean and standard deviation agree with
pyAgrum's exact Gaussian posteriors. The draws are a NUTS estimate (2000
samples), so agreement is asserted at abs=0.1.

Expected runtime: ~5-10 seconds (pyAgrum import + NUTS).
"""

from __future__ import annotations

import pytest

from decisionpy.inference import infer

from ._clg_fixtures import Chain, LongChain, SingleRoot, VStructure

_TOL = 0.1


def _check(fixture, query_vars, observed=None):
    result = infer(fixture.decisionpy(), query_vars, observed=observed or {})
    expected = fixture.pyagrum_query(query_vars, observed=observed)
    for var in query_vars:
        exp_mean, exp_var = expected[var]
        assert result[var].mean() == pytest.approx(exp_mean, abs=_TOL), (
            f"Mean mismatch on {var!r} for {fixture.__class__.__name__}\n"
            f"  observed={observed}\n"
            f"  numpyro:  {result[var].mean()}\n"
            f"  pyagrum:  {exp_mean}"
        )
        assert result[var].std() == pytest.approx(exp_var**0.5, abs=_TOL), (
            f"Std mismatch on {var!r} for {fixture.__class__.__name__}\n"
            f"  observed={observed}\n"
            f"  numpyro:  {result[var].std()}\n"
            f"  pyagrum:  {exp_var**0.5}"
        )


def test_single_root_prior():
    _check(SingleRoot, ["x"])


def test_chain_prior():
    _check(Chain, ["x1", "x2", "x3"])


def test_chain_posterior():
    _check(Chain, ["x1", "x2"], {"x3": 0.4})


def test_vstruct_prior():
    _check(VStructure, ["x1", "x2", "y"])


def test_vstruct_posterior():
    _check(VStructure, ["x1"], {"y": 0.1})


def test_long_chain_prior():
    _check(LongChain, ["a", "b", "c", "d"])


def test_long_chain_deep_evidence():
    _check(LongChain, ["b"], {"d": 2.0, "a": -1.0})
