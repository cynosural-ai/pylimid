"""Tests for :mod:`decisionpy.inference.exact.categorical` — bucket elimination."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference.exact.categorical import solve

# --- helpers ----------------------------------------------------------------


def _single_decision_diagram() -> InfluenceDiagram:
    """
    Single-decision diagram with a hand-computed optimum.

    Nodes: ``disease -> treat`` (info set) ``-> recovery``; utility on
    ``(recovery, treat)``. Hand-solved:
        healthy: no treat → 0.90·100 = 90; treat → 0.92·100 − 20 = 72 → no.
        sick:    no treat → 0.40·100 = 40; treat → 0.80·100 − 20 = 60 → yes.
        E[U*] = 0.6·90 + 0.4·60 = 78.
    """
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="disease",
            states=("healthy", "sick"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
        )
    )
    diag.add_node(
        DecisionNode(name="treat", parents=("disease",), states=("no", "yes"))
    )
    diag.add_node(
        ChanceNode(
            name="recovery",
            parents=("disease", "treat"),
            states=("no", "yes"),
            dist=lambda disease, treat: dist.Categorical(
                probs=jnp.array(
                    [
                        [[0.10, 0.90], [0.08, 0.92]],  # healthy
                        [[0.60, 0.40], [0.20, 0.80]],  # sick
                    ]
                )[disease][treat]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="utility",
            parents=("recovery", "treat"),
            values=lambda recovery, treat: 100.0 * recovery - 20.0 * treat,
        )
    )
    return diag


def _two_decision_diagram() -> InfluenceDiagram:
    """
    Two independent subproblems with additive total utility.

    Subproblem 1: ``s1 -> d1`` (info ``(s1,)``) ``-> o1``; U1 = 100·o1 − 20·d1.
        s1=0: d1=0 → 10, d1=1 → 30 → d1=1.  s1=1: d1=0 → 80, d1=1 → 70 → d1=0.
        E[U1] = 0.6·30 + 0.4·80 = 50.
    Subproblem 2: ``s2 -> d2`` (info ``(s2,)``) ``-> o2``; U2 = 100·o2 − 40·d2.
        d2=0 → 40, d2=1 → 30 → d2=0.  E[U2] = 40.
    Total: 90.
    """
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="s1",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
        )
    )
    diag.add_node(DecisionNode(name="d1", parents=("s1",), states=("a", "b")))
    diag.add_node(
        ChanceNode(
            name="o1",
            parents=("s1", "d1"),
            states=("no", "yes"),
            dist=lambda s1, d1: dist.Categorical(
                probs=jnp.array(
                    [
                        [[0.9, 0.1], [0.5, 0.5]],  # s1=a
                        [[0.2, 0.8], [0.1, 0.9]],  # s1=b
                    ]
                )[s1][d1]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="u1",
            parents=("o1", "d1"),
            values=lambda o1, d1: 100.0 * o1 - 20.0 * d1,
        )
    )
    diag.add_node(
        ChanceNode(
            name="s2",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="d2", parents=("s2",), states=("a", "b")))
    diag.add_node(
        ChanceNode(
            name="o2",
            parents=("d2",),
            states=("no", "yes"),
            dist=lambda d2: dist.Categorical(
                probs=jnp.array([[0.6, 0.4], [0.3, 0.7]])[d2]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="u2",
            parents=("o2", "d2"),
            values=lambda o2, d2: 100.0 * o2 - 40.0 * d2,
        )
    )
    return diag


def _no_information_decision_diagram() -> InfluenceDiagram:
    """
    Decision with an empty information set.

    act=0 → 0.5·100 = 50; act=1 → 0.9·100 − 30 = 60 → act=1.  E[U] = 60.
    """
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="act", states=("no", "yes")))
    diag.add_node(
        ChanceNode(
            name="gain",
            parents=("act",),
            states=("no", "yes"),
            dist=lambda act: dist.Categorical(
                probs=jnp.array([[0.5, 0.5], [0.1, 0.9]])[act]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="utility",
            parents=("gain", "act"),
            values=lambda gain, act: 100.0 * gain - 30.0 * act,
        )
    )
    return diag


def _irrelevant_information_diagram() -> InfluenceDiagram:
    """
    Decision whose information variable does not affect the payoff.

    y is observed but the payoff depends only on the action: act=0 → 100,
    act=1 → 40 → act=0 regardless of y.  E[U] = 100.
    """
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="y",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="act", parents=("y",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="utility",
            parents=("act",),
            values=lambda act: 100.0 - 60.0 * act,
        )
    )
    return diag


# --- single decision ---------------------------------------------------------


def test_single_decision_policy_and_utility() -> None:
    result = solve(_single_decision_diagram().snapshot())
    assert result.policy["treat"] == {(0,): 0, (1,): 1}
    assert result.expected_utility == pytest.approx(78.0)


# --- two decisions -----------------------------------------------------------


def test_two_decisions_independent_information_sets() -> None:
    result = solve(_two_decision_diagram().snapshot())
    assert result.policy["d1"] == {(0,): 1, (1,): 0}
    assert result.policy["d2"] == {(0,): 0, (1,): 0}
    assert result.expected_utility == pytest.approx(90.0)


# --- information-set edge cases ----------------------------------------------


def test_decision_with_empty_information_set() -> None:
    result = solve(_no_information_decision_diagram().snapshot())
    assert result.policy["act"] == {(): 1}
    assert result.expected_utility == pytest.approx(60.0)


def test_irrelevant_information_set() -> None:
    result = solve(_irrelevant_information_diagram().snapshot())
    assert result.policy["act"] == {(0,): 0, (1,): 0}
    assert result.expected_utility == pytest.approx(100.0)


# --- errors ------------------------------------------------------------------


def test_continuous_node_raises() -> None:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0.0, 1.0)))
    diag.add_node(DecisionNode(name="d", states=("a", "b")))
    diag.add_node(UtilityNode(name="u", parents=("d",), values=lambda d: float(d)))
    with pytest.raises(TypeError, match="discrete"):
        solve(diag.snapshot())


def test_non_regular_structure_raises() -> None:
    """A decision whose best action depends on a variable outside its info set."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="structure",
            states=("a", "b"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="test",
            parents=("structure",),
            states=("a", "b"),
            dist=lambda structure: dist.Categorical(
                probs=jnp.array([[0.9, 0.1], [0.2, 0.8]])[structure]
            ),
        )
    )
    diag.add_node(DecisionNode(name="drill", parents=("test",), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="payoff",
            parents=("structure", "drill"),
            values=lambda structure, drill: (
                (100.0 - 80.0 * structure) if drill == 0 else (10.0 + 80.0 * structure)
            ),
        )
    )
    with pytest.raises(NotImplementedError, match="information set"):
        solve(diag.snapshot())
