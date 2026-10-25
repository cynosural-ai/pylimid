"""Public policy-gradient optimization with categorical and continuous information."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    LinearSoftmax,
    TabularPolicy,
    TabularSoftmax,
    UtilityNode,
    solve,
)


def test_solve_trains_declared_categorical_policy_without_method_selection():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode("rain", states=("no", "yes"), dist=lambda: dist.Bernoulli(0.2))
    )
    diagram.add_node(
        DecisionNode(
            "umbrella",
            parents=("rain",),
            states=("no", "yes"),
            policy_family=TabularSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: 1.0 * (rain == umbrella),
        )
    )
    solution = solve(diagram, num_steps=200, num_samples=512)
    rule = solution.policy["umbrella"]
    assert isinstance(rule, TabularPolicy)
    for rain in range(2):
        assert rule.probabilities(rain=jnp.array(rain))[rain] > 0.98
    assert solution.expected_utility > 0.98


def test_solve_accepts_continuous_information_and_explicit_optimizer_settings():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("signal",),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: signal * action,
        )
    )
    solution = solve(diagram, num_steps=300, num_samples=512, learning_rate=0.05)
    assert solution.expected_utility == pytest.approx(0.25, abs=0.04)


def test_solve_requires_a_decision():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Normal(0.0, 1.0)))
    with pytest.raises(InferenceError, match="at least one decision"):
        solve(diagram)
