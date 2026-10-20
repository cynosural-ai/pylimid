"""Analytic gradients, expected utilities, and reuse of learned softmax policies."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    LinearSoftmax,
    SoftmaxPolicy,
    UtilityNode,
    infer,
    solve,
)
from pylimid.numpyro import samples
from pylimid.solver import (
    backward_induction_solve,
    batched_solve,
    policy_gradient_solve,
    scan_solve,
)
from pylimid.solver.policy_gradient import _score_loss


def _signal_diagram():
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
    return diagram


def _uniform_rule(parents=(), parent_states=(), num_features=0):
    return SoftmaxPolicy(
        LinearSoftmax(),
        parents,
        parent_states,
        tuple(() for _ in parents),
        ("off", "on"),
        jnp.zeros((num_features, 2)),
        jnp.zeros(2),
    )


def test_score_gradient_matches_analytic_oracle_and_enumerated_derivative():
    rule = _uniform_rule(("signal",), (None,), 1)
    parameters = {"weights": rule.weights, "bias": rule.bias}
    # Midpoint quadrature over the signal and exact equal weighting of the two
    # actions reproduce the sampling law at zero logits.
    signal = (jnp.arange(4096) + 0.5) / 2048 - 1
    observations = {"signal": jnp.repeat(signal, 2)}
    actions = jnp.tile(jnp.arange(2), len(signal))
    returns = observations["signal"] * actions
    gradients = jax.grad(_score_loss)(
        parameters, rule, observations, actions, returns, 0.0
    )
    assert gradients["weights"] == pytest.approx(
        jnp.array([[1 / 12, -1 / 12]]), abs=1e-6
    )
    assert gradients["bias"] == pytest.approx(jnp.zeros(2), abs=1e-6)

    def enumerated_utility(parameters):
        fitted = replace(rule, **parameters)
        probabilities = jax.vmap(lambda x: fitted.probabilities(signal=x))(signal)
        return jnp.mean(signal * probabilities[:, 1])

    exact = jax.grad(enumerated_utility)(parameters)
    for name in parameters:
        assert gradients[name] == pytest.approx(-exact[name], abs=1e-6)


def test_solver_learns_signal_rule_and_returns_reusable_stochastic_policy():
    diagram = _signal_diagram()
    solution = solve(diagram, method="policy_gradient", num_steps=300, num_samples=512)
    rule = solution.policy["action"]
    assert isinstance(rule, SoftmaxPolicy)
    assert rule.family is diagram["action"].policy_family
    assert rule.parents == ("signal",)
    assert rule.parent_states == (None,)
    assert rule.parent_shapes == ((),)
    assert rule.states == ("off", "on")
    assert rule.weights.shape == (1, 2)
    assert rule.probabilities(signal=jnp.array(-0.8))[0] > 0.95
    assert rule.probabilities(signal=jnp.array(0.8))[1] > 0.95
    assert solution.expected_utility == pytest.approx(0.25, abs=0.035)
    payoff = infer(
        diagram,
        ["payoff"],
        policy=solution.policy,
        num_samples=10000,
        rng_key=jax.random.PRNGKey(22),
    )["payoff"]
    assert payoff.mean() == pytest.approx(0.25, abs=0.02)
    assert solution.method == "policy_gradient"
    assert "stochastic" in solution.render(diagram)


def test_seed_reproduces_fitted_parameters_and_independent_evaluation():
    first = solve(
        _signal_diagram(), method="policy_gradient", num_steps=20, num_samples=64
    )
    second = solve(
        _signal_diagram(), method="policy_gradient", num_steps=20, num_samples=64
    )
    assert isinstance(first.policy["action"], SoftmaxPolicy)
    assert isinstance(second.policy["action"], SoftmaxPolicy)
    assert jnp.array_equal(
        first.policy["action"].weights, second.policy["action"].weights
    )
    assert jnp.array_equal(first.policy["action"].bias, second.policy["action"].bias)
    assert first.expected_utility == second.expected_utility


def test_categorical_features_are_one_hot_with_numerical_features_kept_separate():
    rule = SoftmaxPolicy(
        LinearSoftmax(),
        ("category", "signal"),
        (("red", "green", "blue"), None),
        ((), ()),
        ("off", "on"),
        jnp.array([[0.0, 2.0], [0.0, -2.0], [0.0, 2.0], [0.0, 1.0]]),
        jnp.zeros(2),
    )
    for index, expected in enumerate(
        [jax.nn.sigmoid(2.0), jax.nn.sigmoid(-2.0), jax.nn.sigmoid(2.0)]
    ):
        assert rule.probabilities(signal=jnp.array(0.0), category=jnp.array(index))[
            1
        ] == pytest.approx(expected)
    assert rule.probabilities(signal=jnp.array(1.0), category=jnp.array(0))[
        1
    ] == pytest.approx(jax.nn.sigmoid(3.0))
    with pytest.raises(TypeError, match="Unexpected information"):
        rule.probabilities(
            signal=jnp.array(0.0), category=jnp.array(0), hidden=jnp.array(1.0)
        )


def test_score_solver_handles_discrete_descendants_and_indicator_utilities():
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode("action", states=("off", "on"), policy_family=LinearSoftmax())
    )
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("action",),
            states=("no", "yes"),
            dist=lambda action: dist.Bernoulli(jnp.array([0.1, 0.9])[action]),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("outcome",),
            values=lambda outcome: jnp.where(outcome == 1, 1.0, 0.0),
        )
    )
    solution = solve(diagram, method="policy_gradient", num_steps=200, num_samples=256)
    rule = solution.policy["action"]
    assert isinstance(rule, SoftmaxPolicy)
    assert rule.weights.shape == (0, 2)
    assert rule.probabilities()[1] > 0.95
    assert solution.expected_utility > 0.8


def test_training_uses_mixed_declared_information_and_flattens_vector_values():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode("signal", dist=lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1))
    )
    diagram.add_node(
        ChanceNode(
            "category",
            states=("red", "green", "blue"),
            dist=lambda: dist.Categorical(probs=jnp.ones(3) / 3),
        )
    )
    diagram.add_node(ChanceNode("hidden", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("signal", "category"),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "category", "hidden", "action"),
            values=lambda signal, category, hidden, action: (
                action * (jnp.sum(signal) + jnp.array([-1.0, 1.0, -1.0])[category])
                + 0.1 * hidden
            ),
        )
    )
    solution = solve(diagram, method="policy_gradient", num_steps=300, num_samples=512)
    rule = solution.policy["action"]
    assert isinstance(rule, SoftmaxPolicy)
    assert rule.parents == ("signal", "category")
    assert rule.parent_shapes == ((2,), ())
    assert rule.parent_states == (None, ("red", "green", "blue"))
    assert rule.weights.shape == (5, 2)
    for category, action in [(0, 0), (1, 1), (2, 0)]:
        assert (
            rule.probabilities(signal=jnp.zeros(2), category=jnp.array(category))[
                action
            ]
            > 0.95
        )
    assert solution.expected_utility > 0.4
    with pytest.raises(ValueError, match="must have shape"):
        # The same total feature count cannot conceal a changed parent shape.
        rule.probabilities(signal=jnp.zeros((1, 2)), category=jnp.array(0))


def test_forward_evaluation_preserves_stochastic_action_descendant_correlation():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("action", states=("off", "on")))
    diagram.add_node(DecisionNode("next", parents=("action",), states=("off", "on")))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("action", "next"),
            dist=lambda action, next: dist.Delta(action + 2 * next),
        )
    )
    policy = {"action": _uniform_rule(), "next": {(0,): 1, (1,): 0}}
    draws = samples(diagram.snapshot(), policy=policy, num_samples=1000)
    assert jnp.mean(draws["action"]) == pytest.approx(0.5, abs=0.05)
    assert jnp.array_equal(draws["next"], 1 - draws["action"])
    assert jnp.array_equal(draws["outcome"], 2 - draws["action"])
    with pytest.raises(NotImplementedError, match="forward evaluation"):
        samples(diagram.snapshot(), policy=policy, observed={"outcome": 1})


@pytest.mark.parametrize(
    "changes",
    [
        {"states": ("on", "off")},
        {"parents": ("other",)},
        {"parent_states": (("low", "high"),)},
    ],
)
def test_fitted_policy_rejects_changed_action_or_information_states(changes):
    diagram = _signal_diagram()
    rule = replace(_uniform_rule(("signal",), (None,), 1), **changes)
    with pytest.raises(InferenceError, match="incompatible"):
        samples(diagram.snapshot(), policy={"action": rule})


@pytest.mark.parametrize("method", ["auto", "scan", "backward_induction"])
def test_table_solver_never_silently_replaces_declared_family(method):
    with pytest.raises(InferenceError, match="table solvers cannot"):
        solve(_signal_diagram(), method=method)


@pytest.mark.parametrize(
    "solver", [scan_solve, batched_solve, backward_induction_solve]
)
def test_direct_table_solver_checks_declared_family(solver):
    with pytest.raises(ValueError, match="table solvers cannot"):
        solver(_signal_diagram().snapshot())


def test_solver_requires_family_and_currently_limits_training_to_one_decision():
    diagram = _signal_diagram()
    diagram["action"].policy_family = None
    with pytest.raises(ValueError, match="must declare"):
        solve(diagram, method="policy_gradient")
    diagram.add_node(DecisionNode("other", states=("off", "on")))
    with pytest.raises(NotImplementedError, match="exactly one"):
        solve(diagram, method="policy_gradient")


@pytest.mark.parametrize(
    "options",
    [
        {"num_steps": 0},
        {"num_samples": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
    ],
)
def test_solver_rejects_invalid_optimization_settings(options):
    with pytest.raises(ValueError):
        policy_gradient_solve(_signal_diagram().snapshot(), **options)


def test_optimizer_settings_require_explicit_policy_gradient_method():
    with pytest.raises(ValueError, match="require method"):
        solve(_signal_diagram(), num_steps=10)


def test_family_assignment_rejects_unsupported_representations():
    with pytest.raises(TypeError, match="policy_family"):
        DecisionNode("action", states=("off", "on"), policy_family="network")  # ty: ignore[invalid-argument-type]
