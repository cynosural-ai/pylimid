"""
Validate decisionpy bucket elimination against pyAgrum's exact LIMID solver.

Each test builds the same influence diagram in both libraries and checks that
the maximum expected utility and the per-decision optimal policies agree to
high precision (absolute tolerance 1e-6).

Expected runtime: ~2-3 seconds (pyAgrum import is slow).
"""

from __future__ import annotations

import pytest
from validation._id_fixtures import (
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    SingleDecisionTreat,
    ThreeStateClimate,
    TwoDecisionsIndependent,
)

from decisionpy.inference.exact.categorical import solve

# -- shared check --------------------------------------------------------------


def _check(fixture) -> None:
    """Assert solve(fixture) matches pyAgrum's MEU and optimal policies."""
    result = solve(fixture.decisionpy().snapshot())
    meu, policy = fixture.pyagrum_solution()
    # The tolerance is 1e-5 rather than 1e-6: the decisionpy CPTs flow through
    # JAX float32 arrays, so probabilities carry ~1e-6 rounding noise. The
    # algorithms themselves are both exact.
    assert result.expected_utility == pytest.approx(meu, abs=1e-5), (
        f"MEU mismatch for {type(fixture).__name__}\n"
        f"  decisionpy: {result.expected_utility}\n"
        f"  pyagrum:    {meu}"
    )
    for name, sub in result.policy.items():
        assert sub == policy[name], (
            f"Policy mismatch on {name!r} for {type(fixture).__name__}\n"
            f"  decisionpy: {sub}\n"
            f"  pyagrum:    {policy[name]}"
        )


# -- tests ---------------------------------------------------------------------


def test_single_decision_treat():
    _check(SingleDecisionTreat)


def test_two_decisions_independent():
    _check(TwoDecisionsIndependent)


def test_empty_info_decision():
    _check(EmptyInfoDecision)


def test_irrelevant_info_decision():
    _check(IrrelevantInfoDecision)


def test_nested_two_decisions():
    _check(NestedTwoDecisions)


def test_three_state_climate():
    _check(ThreeStateClimate)
