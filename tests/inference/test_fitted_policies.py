"""
Check fitted-policy inference against small, independently calculable examples.

An LLM constructed these examples and manually derived the reference formulas from
the stated models and Bayes' rule, then wrote the numerical calculations below.
For categorical variables, the calculation lists every combination of states and
actions, multiplies their probabilities by the likelihood of the observed outcome,
and normalizes the weights so they sum to one. For continuous variables, it also
uses Gaussian quadrature to integrate over possible values. NumPy evaluates those
formulas to obtain the reference answers.

The inference implementation produces samples to compare with these answers.
Tolerances allow for sampling error, including correlation between MCMC draws.
Utility checks also verify that values at the same sample index describe the same
scenario. The formulas are written explicitly for human review: agreement on these
small examples provides regression evidence, rather than a proof for every model.
"""

from collections.abc import Mapping
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from pylimid import infer
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode, DecisionNode, UtilityNode
from pylimid.policy import (
    CategoricalPolicy,
    DecisionInformation,
    LinearSoftmax,
    PolicyFamily,
    PolicyParameters,
    TabularSoftmax,
)


class _NonlinearSoftmax(PolicyFamily):
    """Apply a nonlinear transform to scores stored in a nested parameter tree."""

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """Initialize the nested linear layer with zero scores."""
        return {
            "layer": LinearSoftmax().initialize(
                rng_key,
                num_actions=num_actions,
                observations=observations,
                parent_states=parent_states,
            )
        }

    def logits(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """Compute bounded nonlinear action scores for one assignment."""
        return jnp.tanh(
            LinearSoftmax().logits(parameters["layer"], observations, parent_states)
        )


def test_multiple_fitted_policies_match_exact_joint_posterior():
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    left_probs = np.array([0.65, 0.35])
    right_probs = np.array([0.2, 0.3, 0.5])
    first_probs = np.array([[0.2, 0.75, 0.45], [0.8, 0.35, 0.65]])
    diagram.add_node(
        ChanceNode(
            "left", states=labels, dist=lambda: dist.Categorical(probs=left_probs)
        )
    )
    diagram.add_node(
        ChanceNode(
            "right",
            states=("low", "medium", "high"),
            dist=lambda: dist.Categorical(probs=right_probs),
        )
    )
    diagram.add_node(DecisionNode("first", parents=("right", "left"), states=labels))
    diagram.add_node(DecisionNode("second", parents=("left", "first"), states=labels))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("left", "right", "first", "second"),
            states=labels,
            dist=lambda left, right, first, second: dist.Bernoulli(
                0.05 + 0.1 * left + 0.07 * right + 0.2 * first + 0.35 * second
            ),
        )
    )
    diagram.add_node(
        UtilityNode(
            "assignment",
            parents=("left", "right", "first", "second"),
            values=lambda left, right, first, second: (
                ((left * 3 + right) * 2 + first) * 2 + second
            ),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("left", "right", "first", "second"),
            values=lambda left, right, first, second: (
                2 * first - second + 3 * left * first + right
            ),
        )
    )
    policy = {
        "first": CategoricalPolicy(
            TabularSoftmax(),
            ("right", "left"),
            (("low", "medium", "high"), labels),
            ((), ()),
            labels,
            {
                "logits": jnp.log(
                    jnp.stack((1 - first_probs, first_probs), axis=-1).transpose(
                        1, 0, 2
                    )
                )
            },
        ),
        "second": CategoricalPolicy(
            LinearSoftmax(),
            ("left", "first"),
            (labels, labels),
            ((), ()),
            labels,
            {
                "weights": jnp.array([[0.0, 0.0], [0.0, 0.6], [0.0, 0.0], [0.0, 1.1]]),
                "bias": jnp.array([0.0, -0.3]),
            },
        ),
    }

    result = infer(
        diagram,
        ["left", "right", "assignment", "payoff"],
        policy=policy,
        observed={"outcome": 1},
        num_samples=12000,
    )

    weights = np.empty((2, 3, 2, 2))
    for left, right, first, second in product(range(2), range(3), range(2), range(2)):
        first_probability = first_probs[left, right]
        second_probability = 1 / (1 + np.exp(0.3 - 0.6 * left - 1.1 * first))
        weights[left, right, first, second] = (
            left_probs[left]
            * right_probs[right]
            * (first_probability if first else 1 - first_probability)
            * (second_probability if second else 1 - second_probability)
            * (0.05 + 0.1 * left + 0.07 * right + 0.2 * first + 0.35 * second)
        )
    expected = weights / weights.sum()
    assignment = result["assignment"].values
    actual = jnp.bincount(assignment, length=24).reshape(2, 3, 2, 2) / 12000
    assert actual == pytest.approx(expected, abs=0.015)
    assert result["left"].marginal() == pytest.approx(
        expected.sum(axis=(1, 2, 3)), abs=0.025
    )
    assert result["right"].marginal() == pytest.approx(
        expected.sum(axis=(0, 2, 3)), abs=0.025
    )
    first = (assignment // 2) % 2
    second = assignment % 2
    assert jnp.array_equal(assignment // 12, result["left"].values)
    assert jnp.array_equal((assignment // 4) % 3, result["right"].values)
    assert jnp.array_equal(
        result["payoff"].values,
        2 * first - second + 3 * result["left"].values * first + result["right"].values,
    )


@pytest.mark.parametrize(
    "nonlinear", [False, True], ids=["linear-vector", "custom-matrix"]
)
def test_mixed_array_information_posterior_matches_quadrature(nonlinear):
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    shape = (1, 2) if nonlinear else (2,)
    family = _NonlinearSoftmax() if nonlinear else LinearSoftmax()
    diagram.add_node(
        ChanceNode(
            "signal",
            dist=lambda: dist.Normal(jnp.zeros(shape), 1.0).to_event(len(shape)),
        )
    )
    diagram.add_node(
        ChanceNode(
            "alert",
            parents=("signal",),
            states=labels,
            dist=lambda signal: dist.Bernoulli(
                logits=0.7 * jnp.ravel(signal)[0] - 0.25 * jnp.ravel(signal)[1]
            ),
        )
    )
    diagram.add_node(DecisionNode("action", parents=("alert", "signal"), states=labels))
    outcome_probs = jnp.array([[0.1, 0.75], [0.3, 0.9]])
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("alert", "action"),
            states=labels,
            dist=lambda alert, action: dist.Bernoulli(outcome_probs[alert, action]),
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "alert", "action"),
            values=lambda signal, alert, action: (
                jnp.ravel(signal)[0] * action + 0.5 * alert
            ),
        )
    )
    layer = {
        "weights": jnp.array([[0.0, -0.4], [0.0, 0.9], [0.0, 0.8], [0.0, -0.6]]),
        "bias": jnp.array([0.0, -0.1]),
    }
    parameters = {"layer": layer} if nonlinear else layer
    rule = CategoricalPolicy(
        family, ("alert", "signal"), (labels, None), ((), shape), labels, parameters
    )
    before = jax.tree.map(jnp.copy, parameters)

    result = infer(
        diagram,
        ["signal", "alert", "action_value", "payoff"],
        policy={"action": rule},
        observed={"outcome": 1},
        num_samples=5000,
        rng_key=jax.random.PRNGKey(31),
    )

    points, gaussian_weights = _normal_quadrature(40)
    coordinates = np.stack(np.meshgrid(points, points, indexing="ij"), axis=-1)
    weights = gaussian_weights[:, None] * gaussian_weights[None, :]
    alert_probability = 1 / (
        1 + np.exp(-0.7 * coordinates[..., 0] + 0.25 * coordinates[..., 1])
    )
    likelihood = np.zeros_like(weights)
    action_mass = np.zeros_like(weights)
    alert_mass = np.zeros_like(weights)
    payoff_mass = np.zeros_like(weights)
    for alert, action in product(range(2), repeat=2):
        score = (
            [-0.4, 0.9][alert]
            + 0.8 * coordinates[..., 0]
            - 0.6 * coordinates[..., 1]
            - 0.1
        )
        if nonlinear:
            score = np.tanh(score)
        action_probability = 1 / (1 + np.exp(-score))
        mass = (
            (alert_probability if alert else 1 - alert_probability)
            * (action_probability if action else 1 - action_probability)
            * float(outcome_probs[alert, action])
        )
        likelihood += mass
        action_mass += action * mass
        alert_mass += alert * mass
        payoff_mass += (coordinates[..., 0] * action + 0.5 * alert) * mass
    normalizer = np.sum(weights * likelihood)
    expected_signal = (
        np.sum(coordinates * (weights * likelihood)[..., None], axis=(0, 1))
        / normalizer
    )
    assert result["signal"].values.shape == (5000, *shape)
    assert np.asarray(result["signal"].mean()).ravel() == pytest.approx(
        expected_signal, abs=0.08
    )
    assert result["action_value"].mean() == pytest.approx(
        np.sum(weights * action_mass) / normalizer, abs=0.035
    )
    assert result["alert"].marginal()[1] == pytest.approx(
        np.sum(weights * alert_mass) / normalizer, abs=0.04
    )
    assert result["payoff"].mean() == pytest.approx(
        np.sum(weights * payoff_mass) / normalizer, abs=0.08
    )
    assert jnp.allclose(
        result["payoff"].values,
        result["signal"].values.reshape(5000, 2)[:, 0] * result["action_value"].values
        + 0.5 * result["alert"].values,
        atol=1e-6,
    )
    for original, current in zip(
        jax.tree.leaves(before), jax.tree.leaves(parameters), strict=True
    ):
        assert jnp.array_equal(original, current)


@pytest.mark.parametrize(
    "observe_outcome", [False, True], ids=["parent", "parent-and-outcome"]
)
def test_observed_continuous_information_preserves_policy_weights(observe_outcome):
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(DecisionNode("action", parents=("signal",), states=labels))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("action",),
            states=labels,
            dist=lambda action: dist.Bernoulli(jnp.array([0.1, 0.9])[action]),
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: signal * action,
        )
    )
    rule = CategoricalPolicy(
        LinearSoftmax(),
        ("signal",),
        (None,),
        ((),),
        labels,
        {"weights": jnp.array([[0.0, 0.8]]), "bias": jnp.array([0.0, -0.2])},
    )
    observed = {"signal": 0.75}
    if observe_outcome:
        observed["outcome"] = 1

    result = infer(
        diagram,
        ["signal", "action_value", "payoff"],
        policy={"action": rule},
        observed=observed,
        num_samples=6000,
    )

    probability = 1 / (1 + np.exp(-0.4))
    expected = (
        0.9 * probability / (0.1 * (1 - probability) + 0.9 * probability)
        if observe_outcome
        else probability
    )
    assert result["action_value"].mean() == pytest.approx(expected, abs=0.025)
    assert jnp.all(result["signal"].values == 0.75)
    assert jnp.array_equal(
        result["payoff"].values, 0.75 * result["action_value"].values
    )


def test_action_dependent_continuous_information_preserves_joint_posterior():
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    diagram.add_node(DecisionNode("first", states=labels))
    diagram.add_node(
        ChanceNode(
            "signal",
            parents=("first",),
            dist=lambda first: dist.Normal(jnp.array([-1.0, 1.5])[first], 0.7),
        )
    )
    diagram.add_node(DecisionNode("second", parents=("first", "signal"), states=labels))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("signal", "second"),
            states=labels,
            dist=lambda signal, second: dist.Bernoulli(
                logits=0.3 * signal + 1.2 * second - 0.7
            ),
        )
    )
    diagram.add_node(
        UtilityNode("first_value", parents=("first",), values=lambda first: first)
    )
    diagram.add_node(
        UtilityNode("second_value", parents=("second",), values=lambda second: second)
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("first", "signal"),
            values=lambda first, signal: first * signal,
        )
    )
    policy = {
        "first": CategoricalPolicy(
            TabularSoftmax(),
            (),
            (),
            (),
            labels,
            {"logits": jnp.log(jnp.array([0.7, 0.3]))},
        ),
        "second": CategoricalPolicy(
            LinearSoftmax(),
            ("first", "signal"),
            (labels, None),
            ((), ()),
            labels,
            {
                "weights": jnp.array([[0.0, -0.5], [0.0, 0.4], [0.0, 0.8]]),
                "bias": jnp.array([0.0, -0.1]),
            },
        ),
    }

    result = infer(
        diagram,
        ["signal", "first_value", "second_value", "payoff"],
        policy=policy,
        observed={"outcome": 1},
        num_samples=7000,
        rng_key=jax.random.PRNGKey(41),
    )

    points, weights = _normal_quadrature(80)
    normalizer = 0.0
    expected_first = 0.0
    expected_second = 0.0
    expected_signal = 0.0
    expected_payoff = 0.0
    for first, second in product(range(2), repeat=2):
        signal = [-1.0, 1.5][first] + 0.7 * points
        probability = 1 / (1 + np.exp(-([-0.5, 0.4][first] + 0.8 * signal - 0.1)))
        likelihood = 1 / (1 + np.exp(-(0.3 * signal + 1.2 * second - 0.7)))
        mass = (
            [0.7, 0.3][first]
            * (probability if second else 1 - probability)
            * likelihood
        )
        normalizer += np.dot(weights, mass)
        expected_first += first * np.dot(weights, mass)
        expected_second += second * np.dot(weights, mass)
        expected_signal += np.dot(weights, signal * mass)
        expected_payoff += first * np.dot(weights, signal * mass)
    assert result["first_value"].mean() == pytest.approx(
        expected_first / normalizer, abs=0.04
    )
    assert result["second_value"].mean() == pytest.approx(
        expected_second / normalizer, abs=0.035
    )
    assert result["signal"].mean() == pytest.approx(
        expected_signal / normalizer, abs=0.12
    )
    assert result["payoff"].mean() == pytest.approx(
        expected_payoff / normalizer, abs=0.10
    )
    assert jnp.array_equal(
        result["payoff"].values, result["first_value"].values * result["signal"].values
    )


def _normal_quadrature(points):
    """Nodes and normalized weights for standard-normal expectations."""
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
