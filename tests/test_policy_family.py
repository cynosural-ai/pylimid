"""Custom families, nested parameter training, and categorical execution."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    CategoricalPolicy,
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    PolicyFamily,
    UtilityNode,
    infer,
    solve,
)
from pylimid.solver.api import _initialize_policy


class _QuadraticFamily(PolicyFamily):
    """Two action scores depending nonlinearly on a single observed signal."""

    def initialize(self, rng_key, *, num_actions, observations, parent_states):
        assert tuple(observations) == ("signal",)
        assert parent_states == {"signal": None}
        return {"layer": {"weights": 0.01 * jax.random.normal(rng_key, (num_actions,))}}

    def logits(self, parameters, observations, parent_states):
        assert tuple(observations) == ("signal",)
        return parameters["layer"]["weights"] * (observations["signal"] ** 2 - 0.25)


def _diagram(family):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("signal",),
            states=("off", "on"),
            policy_family=family,
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: action * (signal**2 - 0.25),
        )
    )
    return diagram


def test_custom_family_trains_nested_parameters_and_reuses_categorical_execution():
    family = _QuadraticFamily()
    diagram = _diagram(family)
    solution = solve(diagram, num_steps=500, num_samples=512)
    rule = solution.policy["action"]
    assert isinstance(rule, CategoricalPolicy)
    assert rule.family is family
    assert rule.parameters["layer"]["weights"].shape == (2,)
    for signal, action in [(-0.9, 1), (0.0, 0), (0.9, 1)]:
        assert rule.probabilities(signal=jnp.array(signal))[action] > 0.95
    # The optimum acts for |signal| > 1/2; integral_{1/2}^1 (x*x - 1/4) dx = 1/6.
    assert solution.expected_utility == pytest.approx(1 / 6, abs=0.025)
    payoff = infer(
        diagram,
        ["payoff"],
        policy=solution.policy,
        num_samples=10000,
        rng_key=jax.random.PRNGKey(8),
    )["payoff"]
    assert payoff.mean() == pytest.approx(1 / 6, abs=0.02)


def test_initialization_supplies_reproducible_random_keys_to_custom_families():
    snapshot = _diagram(_QuadraticFamily()).snapshot()
    first = _initialize_policy(snapshot, jax.random.PRNGKey(9))["action"]
    repeated = _initialize_policy(snapshot, jax.random.PRNGKey(9))["action"]
    different = _initialize_policy(snapshot, jax.random.PRNGKey(10))["action"]
    weights = first.parameters["layer"]["weights"]
    assert jnp.array_equal(weights, repeated.parameters["layer"]["weights"])
    assert not jnp.array_equal(weights, different.parameters["layer"]["weights"])


@pytest.mark.parametrize("shape", [(), (3,), (1, 2)])
def test_family_logits_must_match_the_decision_action_space(shape):
    class WrongShapeFamily(_QuadraticFamily):
        def logits(self, parameters, observations, parent_states):
            return jnp.zeros(shape)

    with pytest.raises(ValueError, match="Policy logits must have shape"):
        solve(_diagram(WrongShapeFamily()), num_steps=1, num_samples=2)
