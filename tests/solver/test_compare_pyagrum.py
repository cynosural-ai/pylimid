"""
Validate the numpyro solve() against pyAgrum's exact LIMID solver.

Each test solves the same influence diagram in both libraries and checks
that the maximum expected utility agrees within Monte-Carlo noise (the
estimate is a sample mean over 2000 forward passes per policy) and that
the optimal policies agree.

Expected runtime: ~5-10 seconds (pyAgrum import + policy scans).
"""

from __future__ import annotations

from typing import cast

import pyagrum as gum
import pytest

from pylimid import infer, solve
from pylimid.solver.scan.intervention import (
    solve as numpyro_solve,
)

from ._fixtures import (
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
    solution = numpyro_solve(fixture.pylimid().snapshot(), num_samples=num_samples)
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


@pytest.mark.parametrize("treat", [0, 1])
@pytest.mark.parametrize("evidence", [{}, {"disease": 1}, {"recovery": 0}])
def test_utility_under_a_forced_decision(treat, evidence):
    """
    ``infer`` on a utility matches pyAgrum's mean and variance.

    pyAgrum forces an action through evidence on the decision node, which is
    what ``policy=`` does.
    """
    ie = gum.ShaferShenoyLIMIDInference(  # ty: ignore[possibly-missing-attribute]
        SingleDecisionTreat.pyagrum()
    )
    ie.addEvidence("treat", treat)
    for name, value in evidence.items():
        ie.addEvidence(name, value)
    ie.makeInference()
    expected = ie.MEU()

    utility = infer(
        SingleDecisionTreat.pylimid(),
        ["utility"],
        observed=evidence,
        policy={"treat": treat},
        num_samples=20_000,
    )["utility"]
    assert utility.mean() == pytest.approx(expected["mean"], abs=1.5)
    assert utility.std() ** 2 == pytest.approx(expected["variance"], rel=0.05, abs=1.0)


def test_nested_two_decisions():
    """Unreachable D2 cells tie differently; the reachable cells must agree."""
    solution = numpyro_solve(NestedTwoDecisions.pylimid().snapshot(), num_samples=500)
    meu, policy = NestedTwoDecisions.pyagrum_solution()
    assert solution.expected_utility == pytest.approx(meu, abs=5.0)
    assert solution.policy["D1"] == policy["D1"]
    rule = cast(dict[tuple[int, ...], int], solution.policy["D2"])
    for assignment in ((0, 1), (1, 0)):
        assert rule[assignment] == policy["D2"][assignment]


@pytest.mark.parametrize(
    "fixture, utility", [(SingleDecisionTreat, "utility"), (NestedTwoDecisions, "U")]
)
def test_infer_utility_under_solved_policy_matches_pyagrum(fixture, utility):
    solution = solve(fixture.pylimid(), num_samples=1500)
    pyagrum = gum.ShaferShenoyLIMIDInference(  # ty: ignore[possibly-missing-attribute]
        fixture.pyagrum()
    )
    pyagrum.makeInference()
    expected = pyagrum.MEU()
    posterior = infer(
        fixture.pylimid(), [utility], policy=solution.policy, num_samples=20_000
    )[utility]
    assert posterior.mean() == pytest.approx(expected["mean"], abs=1.5)
    assert posterior.std() ** 2 == pytest.approx(
        expected["variance"], rel=0.05, abs=1.0
    )
