"""Categorical table gradients, interactions, and joint training with linear rules."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    CategoricalPolicy,
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    LinearSoftmax,
    TabularSoftmax,
    UtilityNode,
    infer,
    solve,
)
from pylimid._numpyro.samplers import samples
from pylimid.solver.api import _initialize_policy, _score_loss, _trajectory


def _xor_diagram(family):
    diagram = InfluenceDiagram()
    for name in ("left", "right"):
        diagram.add_node(
            ChanceNode(name, states=("no", "yes"), dist=lambda: dist.Bernoulli(0.5))
        )
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("left", "right"),
            states=("off", "on"),
            policy_family=family,
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("left", "right", "action"),
            values=lambda left, right, action: jnp.where(
                action == (left != right), 1.0, 0.0
            ),
        )
    )
    return diagram


def test_tabular_rows_preserve_parent_order_and_have_independent_parameters():
    logits = jnp.arange(18, dtype=float).reshape(3, 2, 3) / 10
    rule = CategoricalPolicy(
        family=TabularSoftmax(),
        parents=("right", "left"),
        parent_states=(("red", "green", "blue"), ("no", "yes")),
        parent_shapes=((), ()),
        states=("low", "medium", "high"),
        parameters={"logits": logits},
    )
    information = {"left": jnp.array(1), "right": jnp.array(2)}
    probabilities = rule.probabilities(**information)
    assert probabilities == pytest.approx(jax.nn.softmax(logits[2, 1]))
    assert jnp.exp(rule.log_prob(jnp.array(2), **information)) == pytest.approx(
        probabilities[2]
    )
    key = jax.random.PRNGKey(4)
    assert rule.sample(key, **information) == rule.sample(key, **information)
    gradient = jax.grad(
        lambda scores: replace(
            rule, parameters={**rule.parameters, "logits": scores}
        ).log_prob(jnp.array(2), **information)
    )(logits)
    assert gradient[2, 1] == pytest.approx(jnp.array([0.0, 0.0, 1.0]) - probabilities)
    assert jnp.count_nonzero(gradient.at[2, 1].set(0)) == 0
    with pytest.raises(TypeError, match="Unexpected information"):
        rule.probabilities(**information, hidden=jnp.array(0))
    with pytest.raises(ValueError, match="must have shape"):
        rule.probabilities(left=jnp.array([1]), right=jnp.array(2))


@pytest.mark.parametrize("nonzero_parameters", [False, True])
def test_tabular_score_gradient_matches_analytic_and_enumerated_expectation(
    nonzero_parameters,
):
    templates = _initialize_policy(
        _xor_diagram(TabularSoftmax()).snapshot(), jax.random.PRNGKey(0)
    )
    rule = templates["action"]
    if nonzero_parameters:
        rule = replace(
            rule,
            parameters={
                **rule.parameters,
                "logits": jnp.array(
                    [[[0.2, -0.3], [0.4, 0.1]], [[-0.5, 0.2], [0.3, -0.2]]]
                ),
            },
        )
    parameters = {"action": {"logits": rule.parameters["logits"]}}
    left, right, action = jnp.indices((2, 2, 2)).reshape(3, -1)
    returns = jnp.where(action == (left != right), 1.0, 0.0)

    def expected_utility(parameters):
        probabilities = jax.nn.softmax(parameters["action"]["logits"], axis=-1)
        return jnp.mean(2 * probabilities[left, right, action] * returns)

    probabilities = jax.nn.softmax(rule.parameters["logits"], axis=-1)[
        left, right, action
    ]
    gradients = jax.grad(_score_loss)(
        parameters,
        templates,
        {"action": {"left": left, "right": right}},
        {"action": action},
        2 * probabilities * returns,
        0.0,
    )
    exact = jax.grad(expected_utility)(parameters)
    assert gradients["action"]["logits"] == pytest.approx(
        -exact["action"]["logits"], abs=1e-7
    )
    if not nonzero_parameters:
        assert gradients["action"]["logits"] == pytest.approx(
            jnp.array([[[-1, 1], [1, -1]], [[1, -1], [-1, 1]]]) / 16,
            abs=1e-7,
        )


def test_tabular_solver_learns_xor_against_analytic_optimum():
    diagram = _xor_diagram(TabularSoftmax())
    solution = solve(diagram, num_steps=200, num_samples=512)
    rule = solution.policy["action"]
    assert isinstance(rule, CategoricalPolicy)
    assert rule.family is diagram["action"].policy_family
    assert rule.parents == ("left", "right")
    assert rule.parent_states == (("no", "yes"), ("no", "yes"))
    assert rule.parent_shapes == ((), ())
    assert rule.states == ("off", "on")
    assert rule.parameters["logits"].shape == (2, 2, 2)
    for left in range(2):
        for right in range(2):
            assert (
                rule.probabilities(left=jnp.array(left), right=jnp.array(right))[
                    int(left != right)
                ]
                > 0.97
            )
    assert solution.expected_utility == pytest.approx(1.0, abs=0.03)
    payoff = infer(diagram, ["payoff"], policy=solution.policy, num_samples=5000)[
        "payoff"
    ]
    assert payoff.mean() > 0.97
    rendered = solution.render()
    assert "TabularSoftmax policy (stochastic)" in rendered
    assert "left=no, right=yes" in rendered
    assert "off=" in rendered and "on=" in rendered


def test_tabular_fitting_is_reproducible_and_evaluated_on_independent_scenarios():
    diagram = _xor_diagram(TabularSoftmax())
    key = jax.random.PRNGKey(9)
    first = solve(diagram, num_steps=20, num_samples=64, rng_key=key)
    second = solve(diagram, num_steps=20, num_samples=64, rng_key=key)
    first_rule, second_rule = first.policy["action"], second.policy["action"]
    assert isinstance(first_rule, CategoricalPolicy)
    assert isinstance(second_rule, CategoricalPolicy)
    assert jnp.array_equal(
        first_rule.parameters["logits"], second_rule.parameters["logits"]
    )
    assert first.expected_utility == second.expected_utility
    information_key, _, _, evaluation_key = jax.random.split(key, 4)
    templates = _initialize_policy(diagram.snapshot(), information_key)
    trajectory = _trajectory(diagram.snapshot(), templates)
    returns, _, _ = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        {"action": {"logits": first_rule.parameters["logits"]}},
        jax.random.split(evaluation_key, 64),
    )
    assert first.expected_utility == float(jnp.mean(returns))


def test_joint_tabular_and_linear_training_with_action_dependent_observations():
    diagram = _xor_diagram(TabularSoftmax())
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(
        ChanceNode(
            "reading",
            parents=("signal", "action"),
            dist=lambda signal, action: dist.Delta(signal + action),
        )
    )
    diagram.add_node(
        DecisionNode(
            "follow_up",
            parents=("reading", "action"),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "later_payoff",
            parents=("signal", "follow_up"),
            values=lambda signal, follow_up: signal * follow_up,
        )
    )
    solution = solve(diagram, num_steps=400, num_samples=1024)
    first, second = solution.policy["action"], solution.policy["follow_up"]
    assert isinstance(first, CategoricalPolicy)
    assert isinstance(second, CategoricalPolicy)
    assert first.family is diagram["action"].policy_family
    assert second.family is diagram["follow_up"].policy_family
    assert second.parents == ("reading", "action")
    assert second.parent_states == (None, ("off", "on"))
    for action in range(2):
        for signal, expected in [(-0.8, 0), (0.8, 1)]:
            assert (
                second.probabilities(
                    reading=jnp.array(signal + action), action=jnp.array(action)
                )[expected]
                > 0.95
            )
    assert solution.expected_utility == pytest.approx(1.25, abs=0.06)
    payoff = infer(
        diagram,
        ["payoff", "later_payoff"],
        policy=solution.policy,
        num_samples=10000,
        rng_key=jax.random.PRNGKey(8),
    )
    total = payoff["payoff"].mean() + payoff["later_payoff"].mean()
    assert total == pytest.approx(1.25, abs=0.04)
    assert solution.render().count("stochastic") == 2


@pytest.mark.parametrize(
    "distribution", [lambda: dist.Normal(0.0, 1.0), lambda: dist.Poisson(2.0)]
)
def test_tabular_family_rejects_information_without_named_states(distribution):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("information", dist=distribution))
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("information",),
            states=("off", "on"),
            policy_family=TabularSoftmax(),
        )
    )
    with pytest.raises(ValueError, match="named finite states.*'information'"):
        solve(diagram)


def test_tabular_family_rejects_array_valued_categorical_information():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            "information",
            states=("no", "yes"),
            dist=lambda: dist.Bernoulli(jnp.full(2, 0.5)),
        )
    )
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("information",),
            states=("off", "on"),
            policy_family=TabularSoftmax(),
        )
    )
    with pytest.raises(ValueError, match="scalar state index.*'information'"):
        solve(diagram)


def test_tabular_root_policy_preserves_stochastic_action_and_descendant_correlation():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("first", states=("off", "on")))
    diagram.add_node(DecisionNode("second", parents=("first",), states=("off", "on")))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("first", "second"),
            dist=lambda first, second: dist.Delta(first + 2 * second),
        )
    )
    rule = CategoricalPolicy(
        TabularSoftmax(),
        (),
        (),
        (),
        ("off", "on"),
        {"logits": jnp.log(jnp.array([0.3, 0.7]))},
    )
    policy = {"first": rule, "second": {(0,): 1, (1,): 0}}
    draws = samples(diagram.snapshot(), policy=policy, num_samples=5000)
    assert jnp.mean(draws["first"]) == pytest.approx(0.7, abs=0.02)
    assert jnp.array_equal(draws["second"], 1 - draws["first"])
    assert jnp.array_equal(draws["outcome"], 2 - draws["first"])
    posterior = samples(
        diagram.snapshot(), policy=policy, observed={"outcome": 1}, num_samples=100
    )
    assert jnp.array_equal(posterior["first"], jnp.ones(100, dtype=int))
    assert jnp.array_equal(posterior["second"], jnp.zeros(100, dtype=int))
    assert jnp.array_equal(posterior["outcome"], jnp.ones(100, dtype=int))
    result = infer(
        diagram, ["outcome"], policy=policy, observed={"outcome": 1}, num_samples=100
    )
    assert jnp.array_equal(result["outcome"].values, posterior["outcome"])


@pytest.mark.parametrize(
    "changes",
    [
        {"states": ("on", "off")},
        {"parents": ("other", "right")},
        {"parent_states": (("yes", "no"), ("no", "yes"))},
    ],
)
def test_tabular_policy_rejects_changed_information_and_action_labels(changes):
    diagram = _xor_diagram(TabularSoftmax())
    rules = _initialize_policy(diagram.snapshot(), jax.random.PRNGKey(0))
    rules["action"] = replace(rules["action"], **changes)
    with pytest.raises(InferenceError, match="incompatible"):
        samples(diagram.snapshot(), policy=rules)
