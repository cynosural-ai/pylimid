"""Analytic score gradients and end-to-end continuous decision optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from pylimid import (
    CategoricalPolicy,
    ChanceNode,
    ContinuousPolicy,
    DecisionNode,
    DistributionPolicy,
    InfluenceDiagram,
    Interval,
    Linear,
    LinearSoftmax,
    Real,
    UtilityNode,
    infer,
    solve,
)
from pylimid.solver.api import _initialize_policy, _score_loss, _trajectory


def _normal_rule():
    family = DistributionPolicy(dist.Normal, Linear())
    return ContinuousPolicy(
        family,
        (),
        (),
        (),
        Real(),
        {"weights": jnp.empty((0, 2)), "bias": jnp.array([0.3, jnp.log(0.7)])},
    )


def test_normal_score_gradient_matches_closed_form():
    rule = _normal_rule()
    points, weights = np.polynomial.hermite.hermgauss(48)
    weights = jnp.asarray(weights / np.sqrt(np.pi))
    actions = 0.3 + 0.7 * jnp.asarray(points * np.sqrt(2.0))
    returns = -((actions - 1.2) ** 2)
    gradients = jax.grad(_score_loss)(
        {"action": rule.parameters},
        {"action": rule},
        {"action": {}},
        {"action": actions},
        len(points) * weights * returns,
        0.0,
    )
    # E[-(A-t)^2] = -(mu-t)^2-sigma^2. The loss has the opposite derivative.
    assert gradients["action"]["bias"] == pytest.approx(
        jnp.array([2 * (0.3 - 1.2), 2 * 0.7**2]), abs=2e-6
    )


def test_continuous_action_with_bernoulli_descendant_has_correct_score_gradient():
    rule = _normal_rule()
    points, weights = np.polynomial.hermite.hermgauss(48)
    points = jnp.asarray(points * np.sqrt(2.0))
    weights = jnp.asarray(weights / np.sqrt(np.pi))
    actions = 0.3 + 0.7 * points
    p = jax.nn.sigmoid(actions)
    branch_actions = jnp.repeat(actions, 2)
    outcomes = jnp.tile(jnp.array([0.0, 1.0]), len(points))
    branch_weights = (weights[:, None] * jnp.stack((1 - p, p), axis=-1)).ravel()
    gradients = jax.grad(_score_loss)(
        {"action": rule.parameters},
        {"action": rule},
        {"action": {}},
        {"action": branch_actions},
        len(branch_actions) * branch_weights * (outcomes - 0.1 * branch_actions**2),
        0.0,
    )

    def loss(parameters):
        bias = parameters["bias"]
        a = bias[0] + jnp.exp(bias[1]) * points
        return -jnp.sum(weights * (jax.nn.sigmoid(a) - 0.1 * a**2))

    expected = jax.grad(loss)(rule.parameters)
    assert gradients["action"]["bias"] == pytest.approx(expected["bias"], abs=2e-6)


def test_mixed_decision_gradient_matches_exact_branching_and_gaussian_quadrature():
    first = _normal_rule()
    second = CategoricalPolicy(
        LinearSoftmax(),
        ("reading",),
        (None,),
        ((),),
        ("off", "on"),
        {"weights": jnp.array([[0.0, 0.8]]), "bias": jnp.array([0.0, -0.2])},
    )
    templates = {"first": first, "second": second}
    parameters = {name: rule.parameters for name, rule in templates.items()}
    points, weights = np.polynomial.hermite.hermgauss(48)
    points = jnp.asarray(points * np.sqrt(2.0))
    weights = jnp.asarray(weights / np.sqrt(np.pi))

    def values(parameters):
        bias = parameters["first"]["bias"]
        a = bias[0] + jnp.exp(bias[1]) * points
        reading = 2 * a + 0.1
        b = parameters["second"]
        p = jax.nn.sigmoid(
            reading * (b["weights"][0, 1] - b["weights"][0, 0])
            + b["bias"][1]
            - b["bias"][0]
        )
        return a, reading, p

    a, reading, p = values(parameters)
    mass = (weights[:, None] * jnp.stack((1 - p, p), axis=-1)).ravel()
    first_actions = jnp.repeat(a, 2)
    second_actions = jnp.tile(jnp.array([0, 1]), len(points))
    returns = first_actions * second_actions - 0.1 * first_actions**2
    gradients = jax.grad(_score_loss)(
        parameters,
        templates,
        {"first": {}, "second": {"reading": jnp.repeat(reading, 2)}},
        {"first": first_actions, "second": second_actions},
        len(mass) * mass * returns,
        0.0,
    )

    def loss(parameters):
        a, _, p = values(parameters)
        return -jnp.sum(weights * (a * p - 0.1 * a**2))

    expected = jax.grad(loss)(parameters)
    for actual, reference in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(expected), strict=True
    ):
        assert actual == pytest.approx(reference, abs=3e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_continuous_policy_learns_linear_response_and_reuses_it_for_inference(seed):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    family = DistributionPolicy(dist.Normal, Linear(), fixed_parameters={"scale": 0.2})
    diagram.add_node(
        DecisionNode(
            "action", parents=("signal",), action_space=Real(), policy_family=family
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: -((action - (0.4 + 0.6 * signal)) ** 2),
        )
    )
    solution = solve(
        diagram,
        num_steps=180,
        num_samples=512,
        learning_rate=0.03,
        rng_key=jax.random.PRNGKey(seed),
    )
    rule = solution.policy["action"]
    assert isinstance(rule, ContinuousPolicy)
    assert rule.parameters["bias"][0] == pytest.approx(0.4, abs=0.04)
    assert rule.parameters["weights"][0, 0] == pytest.approx(0.6, abs=0.05)
    assert solution.expected_utility == pytest.approx(-(0.2**2), abs=0.012)
    payoff = infer(diagram, ["payoff"], policy=solution.policy, num_samples=5000)[
        "payoff"
    ]
    assert payoff.mean() == pytest.approx(-(0.2**2), abs=0.012)
    assert "continuous action space" in solution.render()


def test_bounded_beta_training_respects_the_domain():
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode(
            "action",
            action_space=Interval(0.0, 1.0),
            policy_family=DistributionPolicy(dist.Beta, Linear()),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff", parents=("action",), values=lambda action: -((action - 0.7) ** 2)
        )
    )
    solution = solve(diagram, num_steps=200, num_samples=512, learning_rate=0.03)
    rule = solution.policy["action"]
    assert isinstance(rule, ContinuousPolicy)
    assert rule.distribution().mean == pytest.approx(0.7, abs=0.07)
    assert solution.expected_utility > -0.03
    draws = jax.vmap(rule.sample)(jax.random.split(jax.random.PRNGKey(2), 1024))
    assert jnp.all((draws >= 0) & (draws <= 1))


def test_fixed_stochastic_family_needs_no_trainable_outputs():
    diagram = InfluenceDiagram()
    family = DistributionPolicy(
        dist.Normal, Linear(outputs=()), fixed_parameters={"loc": 0.5, "scale": 0.2}
    )
    diagram.add_node(DecisionNode("action", action_space=Real(), policy_family=family))
    diagram.add_node(
        UtilityNode("payoff", parents=("action",), values=lambda action: action)
    )
    solution = solve(diagram, num_steps=1, num_samples=2000)
    rule = solution.policy["action"]
    assert isinstance(rule, ContinuousPolicy)
    assert rule.parameters["bias"].shape == (0,)
    assert solution.expected_utility == pytest.approx(0.5, abs=0.02)


def test_initialization_binds_a_continuous_action_as_later_categorical_information():
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode(
            "power",
            action_space=Real(),
            policy_family=DistributionPolicy(dist.Normal, Linear()),
        )
    )
    diagram.add_node(
        DecisionNode(
            "switch",
            parents=("power",),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    snapshot = diagram.snapshot()
    rules = _initialize_policy(snapshot, jax.random.PRNGKey(0))
    assert isinstance(rules["power"], ContinuousPolicy)
    assert isinstance(rules["switch"], CategoricalPolicy)
    assert rules["switch"].parent_states == (None,)


def test_mixed_decisions_train_together_with_action_dependent_continuous_information():
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode("mode", states=("off", "on"), policy_family=LinearSoftmax())
    )
    diagram.add_node(
        ChanceNode(
            "signal", parents=("mode",), dist=lambda mode: dist.Normal(0.5 * mode, 0.2)
        )
    )
    diagram.add_node(
        DecisionNode(
            "power",
            parents=("mode", "signal"),
            action_space=Real(),
            policy_family=DistributionPolicy(
                dist.Normal, Linear(), fixed_parameters={"scale": 0.2}
            ),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("mode", "signal", "power"),
            values=lambda mode, signal, power: (
                0.5 * mode - (power - signal - 0.2 * mode) ** 2
            ),
        )
    )
    solution = solve(diagram, num_steps=300, num_samples=512, learning_rate=0.03)
    mode, power = solution.policy["mode"], solution.policy["power"]
    assert isinstance(mode, CategoricalPolicy)
    assert isinstance(power, ContinuousPolicy)
    assert mode.probabilities()[1] > 0.95
    assert power.distribution(
        mode=jnp.array(1), signal=jnp.array(0.5)
    ).mean == pytest.approx(0.7, abs=0.06)
    assert solution.expected_utility > 0.4
    assert (
        infer(diagram, ["payoff"], policy=solution.policy, num_samples=5000)[
            "payoff"
        ].mean()
        > 0.4
    )


def test_complete_trajectories_sample_action_dependent_bernoulli_outcomes():
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode(
            "action",
            action_space=Real(),
            policy_family=DistributionPolicy(dist.Normal, Linear()),
        )
    )
    diagram.add_node(
        ChanceNode(
            "success",
            parents=("action",),
            states=("no", "yes"),
            dist=lambda action: dist.Bernoulli(logits=action),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("action", "success"),
            values=lambda action, success: success - 0.1 * action**2,
        )
    )
    snapshot = diagram.snapshot()
    templates = _initialize_policy(snapshot, jax.random.PRNGKey(0))
    parameters = {"action": _normal_rule().parameters}
    trajectory = _trajectory(snapshot, templates)
    batch = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        parameters, jax.random.split(jax.random.PRNGKey(7), 30000)
    )
    points, weights = np.polynomial.hermite.hermgauss(64)
    actions = 0.3 + 0.7 * np.sqrt(2) * points
    expected = (weights @ (1 / (1 + np.exp(-actions)) - 0.1 * actions**2)) / np.sqrt(
        np.pi
    )
    assert jnp.mean(batch.utility) == pytest.approx(expected, abs=0.015)


def test_truncated_normal_policy_trains_with_fixed_bounds():
    diagram = InfluenceDiagram()
    family = DistributionPolicy(
        dist.TruncatedNormal,
        Linear(),
        fixed_parameters={"low": -1.0, "high": 1.0, "scale": 0.2},
    )
    diagram.add_node(
        DecisionNode("action", action_space=Interval(-1.0, 1.0), policy_family=family)
    )
    diagram.add_node(
        UtilityNode(
            "payoff", parents=("action",), values=lambda action: -((action - 0.3) ** 2)
        )
    )
    solution = solve(diagram, num_steps=180, num_samples=512, learning_rate=0.03)
    rule = solution.policy["action"]
    assert isinstance(rule, ContinuousPolicy)
    assert rule.distribution().mean == pytest.approx(0.3, abs=0.04)
    assert solution.expected_utility == pytest.approx(-0.04, abs=0.012)
