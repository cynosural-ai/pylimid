"""
Validate backward induction against pyAgrum's exact LIMID solver and the scan.

Backward induction must return an optimal policy — the same one pyAgrum's
exact solver finds, or one with the same expected utility when optimal
policies tie — and agree with the global scan at Monte-Carlo tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from pylimid.inference.numpyro.solvers import backward_induction_solve
from pylimid.inference.numpyro.solvers.intervention_scan import solve as scan_solve

from .._limid_fixtures import (
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    SingleDecisionTreat,
    ThreeStateClimate,
    TwoDecisionsIndependent,
)

FIXTURES = {
    "SingleDecisionTreat": SingleDecisionTreat,
    "TwoDecisionsIndependent": TwoDecisionsIndependent,
    "EmptyInfoDecision": EmptyInfoDecision,
    "IrrelevantInfoDecision": IrrelevantInfoDecision,
    "NestedTwoDecisions": NestedTwoDecisions,
    "ThreeStateClimate": ThreeStateClimate,
}

#: Sample-mean noise of the MEU estimate at 2000 samples: utilities scale
#: to ~100, so the error stays within a few units.
_MEU_TOL = 3.0


@pytest.mark.parametrize("fixture", FIXTURES.values(), ids=list(FIXTURES))
def test_matches_pyagrum(fixture):
    solution = backward_induction_solve(fixture.pylimid().snapshot())
    meu, policy = fixture.pyagrum_solution()
    assert solution.expected_utility == pytest.approx(meu, abs=_MEU_TOL)
    assert solution.policy == policy


@pytest.mark.parametrize("fixture", FIXTURES.values(), ids=list(FIXTURES))
def test_matches_the_scan(fixture):
    """The scan is the global reference; both are Monte-Carlo estimates."""
    snapshot = fixture.pylimid().snapshot()
    ours = backward_induction_solve(snapshot, num_samples=500)
    scan = scan_solve(snapshot, num_samples=500)
    assert ours.expected_utility == pytest.approx(scan.expected_utility, abs=9.0)


def test_same_seed_reproduces_the_solution():
    snapshot = NestedTwoDecisions.pylimid().snapshot()
    first = backward_induction_solve(
        snapshot, num_samples=500, rng_key=jax.random.PRNGKey(3)
    )
    second = backward_induction_solve(
        snapshot, num_samples=500, rng_key=jax.random.PRNGKey(3)
    )
    assert first == second


def test_non_solvable_raises():
    with pytest.raises(ValueError, match="not solvable"):
        backward_induction_solve(_shared_utility_diagram().snapshot())


def test_higher_level_unobserved_influence_raises():
    """
    Raise on a higher-level unobserved influence on a decision's utilities.

    The diagram is unsolvable even though pyAgrum's level-based gate admits
    it and returns a suboptimal policy.
    """
    with pytest.raises(ValueError, match="not solvable"):
        backward_induction_solve(_higher_level_shared_utility_diagram().snapshot())


def test_higher_level_unobserved_influence_scan_finds_the_optimum():
    """The scan coordinates both decisions: both play a, EU 5."""
    solution = scan_solve(
        _higher_level_shared_utility_diagram().snapshot(), num_samples=500
    )
    assert solution.expected_utility == pytest.approx(5.0)
    assert solution.policy == {"D1": {(): 0}, "D2": {(0,): 0, (1,): 0}}


def test_continuous_information_set_raises():
    with pytest.raises(ValueError, match="continuous"):
        backward_induction_solve(_continuous_info_diagram().snapshot())


# ---------------------------------------------------------------------------
# Diagrams for the failure paths
# ---------------------------------------------------------------------------


def _shared_utility_diagram() -> InfluenceDiagram:
    """A shared utility with disjoint info sets: no ordering exists."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="X",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="D1", parents=("X",), states=("a", "b")))
    diag.add_node(DecisionNode(name="D2", parents=("X",), states=("a", "b")))
    diag.add_node(
        UtilityNode(
            name="U", parents=("D1", "D2"), values=lambda D1, D2: (D1 == D2) * 1.0
        )
    )
    return diag


def _continuous_info_diagram() -> InfluenceDiagram:
    """A decision observing a continuous parent: an untabulated info set."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
    diag.add_node(DecisionNode(name="d", parents=("x",), states=("no", "yes")))
    diag.add_node(UtilityNode(name="u", parents=("x", "d"), values=lambda x, d: x * d))
    return diag


def _higher_level_shared_utility_diagram() -> InfluenceDiagram:
    """
    A higher-level unobserved influence on a shared utility.

    ``D1 -> X -> D2`` with both decisions feeding one utility, and ``X``
    uninformative about ``D1``: the levels admit the diagram (pyAgrum
    returns 4.0), but one-pass backward induction is unsound — the
    coordinated policy both play ``a`` scores 5.0.
    """
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="D1", states=("a", "b")))
    diag.add_node(
        ChanceNode(
            name="X",
            parents=("D1",),
            states=("a", "b"),
            dist=lambda D1: dist.Categorical(
                probs=jnp.array([[0.5, 0.5], [0.5, 0.5]])[D1]
            ),
        )
    )
    diag.add_node(DecisionNode(name="D2", parents=("X",), states=("a", "b")))
    diag.add_node(
        UtilityNode(
            name="U",
            parents=("D1", "D2"),
            values=lambda D1, D2: jnp.array([[5.0, 4.0], [0.0, 3.0]])[D1, D2],
        )
    )
    return diag
