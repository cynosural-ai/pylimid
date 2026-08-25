"""
Validate the numpyro solve() against pyAgrum's exact LIMID solver.

Each test solves the same influence diagram in both libraries and checks
that the maximum expected utility agrees within Monte-Carlo noise (the
estimate is a sample mean over 2000 forward passes per policy) and that
the optimal policies agree.

Expected runtime: ~5-10 seconds (pyAgrum import + policy scans).
"""

from __future__ import annotations

import pytest

from decisionpy.inference.numpyro.solver import solve as numpyro_solve

from ._limid_fixtures import (
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    SingleDecisionTreat,
    ThreeStateClimate,
    TwoDecisionsIndependent,
)

#: Sample-mean noise of the MEU estimate: utilities scale to ~100 and the
#: estimate is a mean over forward passes, so the error stays within a few
#: units at 2000 samples and about twice that at 500.
_MEU_TOL = 3.0


def _check(fixture, *, num_samples: int = 2000, meu_tol: float = _MEU_TOL):
    solution = numpyro_solve(fixture.decisionpy().snapshot(), num_samples=num_samples)
    meu, policy = fixture.pyagrum_solution()
    assert solution.expected_utility == pytest.approx(meu, abs=meu_tol), (
        f"MEU mismatch for {fixture.__class__.__name__}: "
        f"numpyro={solution.expected_utility}, pyagrum={meu}"
    )
    assert solution.policy == policy, (
        f"Policy mismatch for {fixture.__class__.__name__}: "
        f"numpyro={solution.policy}, pyagrum={policy}"
    )


def test_single_decision_treat():
    _check(SingleDecisionTreat)


def test_two_decisions_independent():
    # 16-policy scan; fewer samples keep the comparison suite fast.
    _check(TwoDecisionsIndependent, num_samples=500, meu_tol=5.0)


def test_empty_info_decision():
    _check(EmptyInfoDecision)


def test_irrelevant_info_decision():
    _check(IrrelevantInfoDecision)


def test_three_state_climate():
    _check(ThreeStateClimate)


def test_nested_two_decisions():
    """Unreachable D2 cells tie differently; the reachable cells must agree."""
    solution = numpyro_solve(
        NestedTwoDecisions.decisionpy().snapshot(), num_samples=500
    )
    meu, policy = NestedTwoDecisions.pyagrum_solution()
    assert solution.expected_utility == pytest.approx(meu, abs=5.0)
    assert solution.policy["D1"] == policy["D1"]
    for assignment in ((0, 1), (1, 0)):
        assert solution.policy["D2"][assignment] == policy["D2"][assignment]
