"""
Validate backward induction against pyAgrum's exact LIMID solver and the scan.

Backward induction must return an optimal policy — the same one pyAgrum's
exact solver finds, or one with the same expected utility when optimal
policies tie — and agree with the global scan at Monte-Carlo tolerance.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import solve
from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from pylimid.solver import backward_induction_solve
from pylimid.solver.scan.intervention import solve as scan_solve

from ._fixtures import (
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
    with pytest.raises(ValueError, match="without named states"):
        backward_induction_solve(_continuous_info_diagram().snapshot())


@pytest.mark.parametrize("method", [None, "auto", "backward_induction"])
def test_impossible_parent_combinations_warn_and_return_a_complete_policy(method):
    diagram = _inspection_diagram()
    with pytest.warns(UserWarning, match="Decision 'buy'") as captured:
        if method is None:
            solution = backward_induction_solve(diagram.snapshot(), num_samples=1000)
        else:
            solution = solve(diagram, method=method, num_samples=1000)

    assert solution.method == "backward_induction"
    assert solution.policy == {
        "test": {(): 1},
        "buy": {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (1, 0): 0,
            (1, 1): 1,
            (1, 2): 0,
        },
    }
    assert solution.expected_utility == pytest.approx(4.0, abs=0.6)
    assert len(captured) == 1
    message = str(captured[0].message)
    assert (
        "[{'test': 'no_test', 'result': 'success'}, "
        "{'test': 'no_test', 'result': 'failure'}, "
        "{'test': 'test', 'result': 'no_result'}]"
    ) in message
    assert "1000 trajectories" in message
    assert "first action, index 0 ('do_not_buy')" in message
    assert "not optimised" in message
    assert "rare or impossible" in message


def test_a_rare_unsampled_state_warns_and_uses_a_reproducible_fallback():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="signal",
            states=("common", "rare"),
            dist=lambda: dist.Categorical(probs=jnp.array([1.0 - 1e-9, 1e-9])),
        )
    )
    diagram.add_node(
        DecisionNode(name="act", parents=("signal",), states=("first", "second"))
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("signal", "act"),
            values=lambda signal, act: jnp.array([[-2.0, -1.0], [-10.0, 10.0]])[
                signal, act
            ],
        )
    )
    snapshot = diagram.snapshot()
    with pytest.warns(UserWarning, match="'signal': 'rare'") as first_warning:
        first = backward_induction_solve(snapshot, num_samples=128)
    with pytest.warns(UserWarning, match="'signal': 'rare'") as second_warning:
        second = backward_induction_solve(snapshot, num_samples=128)

    assert first == second
    assert first.policy == {"act": {(0,): 1, (1,): 0}}
    assert first.expected_utility == pytest.approx(-1.0)
    assert len(first_warning) == len(second_warning) == 1
    message = str(first_warning[0].message)
    assert message == str(second_warning[0].message)
    assert "'signal': 'common'" not in message
    assert "num_samples can help only if they are possible" in message


@pytest.mark.parametrize("fixture", [EmptyInfoDecision, SingleDecisionTreat])
def test_observed_parent_combinations_do_not_warn(fixture):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        backward_induction_solve(fixture.pylimid().snapshot())


# ---------------------------------------------------------------------------
# Diagrams for the failure paths
# ---------------------------------------------------------------------------


def _inspection_diagram() -> InfluenceDiagram:
    """Buy after a successful test; skipping the test produces no result."""
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode(name="test", states=("no_test", "test")))
    diagram.add_node(
        ChanceNode(
            name="result",
            parents=("test",),
            states=("no_result", "success", "failure"),
            dist=lambda test: dist.Categorical(
                probs=jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])[test]
            ),
        )
    )
    diagram.add_node(
        DecisionNode(
            name="buy", parents=("test", "result"), states=("do_not_buy", "buy")
        )
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("test", "result", "buy"),
            values=lambda test, result, buy: (
                jnp.where(buy == 1, jnp.where(result == 1, 10.0, -10.0), 0.0)
                + 2.0 * (1 - test)
                - test
            ),
        )
    )
    return diagram


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
