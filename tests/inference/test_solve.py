"""Tests for :func:`pylimid.inference.solve` — the solver front door."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from pylimid.inference import InferenceError, solve
from pylimid.inference.numpyro.solvers import (
    backward_induction_solve,
    batched_solve,
    scan_solve,
)

# --- helpers -----------------------------------------------------------------


def _umbrella_diagram() -> InfluenceDiagram:
    """``rain -> umbrella (decision)``; utility on ``(rain, umbrella)``."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        DecisionNode(name="umbrella", parents=("rain",), states=("no", "yes"))
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: (rain == umbrella) * 1.0,
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


def _mixed_diagram() -> InfluenceDiagram:
    """Discrete signal, continuous outcome: s -> x | s -> d(s) -> U."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="s",
            states=("low", "high"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.3, 0.7])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="x",
            parents=("s",),
            dist=lambda s: dist.Normal(loc=-1.0 + 2.0 * s, scale=1.0),
        )
    )
    diag.add_node(DecisionNode(name="d", parents=("s",), states=("no", "yes")))
    diag.add_node(UtilityNode(name="u", parents=("x", "d"), values=lambda x, d: x * d))
    return diag


def _non_solvable_diagram() -> InfluenceDiagram:
    """
    ``D1 -> X -> D2`` with a shared utility: no backward-induction order.

    ``X`` is uninformative about ``D1``, so ``D2`` cannot be resolved without
    knowing ``D1``'s rule. The coordinated policy (both play ``a``) scores 5.
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


def _bayesian_network() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    return diag


# --- solving -----------------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "backward_induction", "scan"])
def test_solve_policy_is_optimal(method):
    """Umbrella should mirror rain: {(0,): 0, (1,): 1} with E[U] = 1.0."""
    solution = solve(_umbrella_diagram(), method=method)
    assert solution.policy == {"umbrella": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("method", ["auto", "backward_induction", "scan"])
def test_solve_mixed_diagram_solves(method):
    """
    Continuous chance nodes do not rule the solvers out.

    E[x | s=0] = -1 → d=0; E[x | s=1] = 1 → d=1; E[U] = 0.7.
    """
    solution = solve(_mixed_diagram(), method=method)
    assert solution.policy == {"d": {(0,): 0, (1,): 1}}
    assert solution.expected_utility == pytest.approx(0.7, abs=0.15)


def test_auto_uses_backward_induction_on_a_solvable_diagram():
    diag = _mixed_diagram()
    solution = solve(diag)
    assert solution == backward_induction_solve(diag.snapshot())
    assert solution.method == "backward_induction"


@pytest.mark.parametrize("method", ["backward_induction", "scan"])
def test_explicit_method_is_recorded(method):
    assert solve(_umbrella_diagram(), method=method).method == method


def test_scan_method_is_the_batched_scan():
    diag = _mixed_diagram()
    assert solve(diag, method="scan") == batched_solve(diag.snapshot())


def test_auto_falls_back_to_the_scan_on_a_non_solvable_diagram():
    diag = _non_solvable_diagram()
    solution = solve(diag, num_samples=500)
    assert solution == batched_solve(diag.snapshot(), num_samples=500)
    assert solution.method == "scan"
    assert solution.policy == {"D1": {(): 0}, "D2": {(0,): 0, (1,): 0}}
    assert solution.expected_utility == pytest.approx(5.0)


def test_backward_induction_on_a_non_solvable_diagram_raises():
    with pytest.raises(InferenceError, match="not solvable"):
        solve(_non_solvable_diagram(), method="backward_induction")


def test_num_samples_and_rng_key_are_forwarded():
    diag = _mixed_diagram()
    key = jax.random.PRNGKey(7)
    assert solve(diag, method="scan", num_samples=300, rng_key=key) == batched_solve(
        diag.snapshot(), num_samples=300, rng_key=key
    )


def test_solve_is_a_monte_carlo_estimate():
    """The MEU is a sample estimate; it wobbles with the seed."""
    estimates = {
        solve(_mixed_diagram(), rng_key=jax.random.PRNGKey(seed)).expected_utility
        for seed in range(3)
    }
    assert len(estimates) >= 2  # different seeds, different estimates
    assert all(value == pytest.approx(0.7, abs=0.15) for value in estimates)


def test_unbatched_scan_agrees_with_solve():
    """The unbatched scan stays importable as the reference implementation."""
    diag = _umbrella_diagram()
    reference = scan_solve(diag.snapshot())
    assert reference.policy == solve(diag, method="scan").policy
    assert reference.method == "scan"


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        solve(_umbrella_diagram(), method="exact")  # ty: ignore[invalid-argument-type]


def test_solve_no_decisions_raises():
    with pytest.raises(InferenceError, match="at least one decision"):
        solve(_bayesian_network())


@pytest.mark.parametrize("method", ["auto", "backward_induction", "scan"])
def test_solve_continuous_information_set_raises(method):
    with pytest.raises(InferenceError, match="continuous"):
        solve(_continuous_info_diagram(), method=method)


# --- rendering ---------------------------------------------------------------


def test_render_labels_policy_with_node_states():
    diag = _umbrella_diagram()
    assert solve(diag).render(diag) == (
        "expected utility: 1.00 (method: backward_induction)\n"
        "\n"
        "umbrella (rain)\n"
        "  rain=no  -> no\n"
        "  rain=yes -> yes"
    )


def test_render_multiple_parents_aligns_situations():
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="a",
            states=("x", "yy"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="b",
            states=("0", "1"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="d", parents=("a", "b"), states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="u",
            parents=("d", "a", "b"),
            values=lambda d, a, b: (d == a * b) * 1.0,
        )
    )
    assert solve(diag).render(diag) == (
        "expected utility: 1.00 (method: backward_induction)\n"
        "\n"
        "d (a, b)\n"
        "  a=x, b=0  -> no\n"
        "  a=x, b=1  -> no\n"
        "  a=yy, b=0 -> no\n"
        "  a=yy, b=1 -> yes"
    )


def test_render_no_parents_situation_is_always():
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d", states=("a", "b")))
    diag.add_node(UtilityNode(name="u", parents=("d",), values=lambda d: d * 1.0))
    assert solve(diag).render(diag) == (
        "expected utility: 1.00 (method: backward_induction)\n\nd (no parents)\n"
        "  always -> b"
    )
