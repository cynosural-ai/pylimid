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
    TabularPolicy,
    TabularSoftmax,
    UtilityNode,
    infer,
    solve,
)
from pylimid.numpyro import samples
from pylimid.solver import (
    backward_induction_solve,
    batched_solve,
    is_solvable,
    policy_gradient_solve,
    scan_solve,
)
from pylimid.solver.policy_gradient import (
    _initialize_policy,
    _score_loss,
    _trajectory,
)


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


def _sequential_diagram(remember_first=False):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(
        DecisionNode(
            "first",
            parents=("signal",),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        ChanceNode(
            "reading",
            parents=("signal", "first"),
            dist=lambda signal, first: dist.Delta(signal + first),
        )
    )
    diagram.add_node(
        DecisionNode(
            "second",
            parents=("reading", "first") if remember_first else ("reading",),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "early_payoff",
            parents=("signal", "first"),
            values=lambda signal, first: signal * first,
        )
    )
    diagram.add_node(
        UtilityNode(
            "late_payoff",
            parents=("signal", "first", "second"),
            values=lambda signal, first, second: second * (2 * signal + first),
        )
    )
    return diagram


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
        {"action": parameters},
        {"action": rule},
        {"action": observations},
        {"action": actions},
        returns,
        0.0,
    )["action"]
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


@pytest.mark.parametrize("nonzero_parameters", [False, True])
def test_joint_score_gradient_matches_exact_action_branching(nonzero_parameters):
    snapshot = _sequential_diagram().snapshot()
    assert not is_solvable(snapshot)
    templates = _initialize_policy(snapshot, jax.random.PRNGKey(1))
    parameters = {
        name: {"weights": rule.weights, "bias": rule.bias}
        for name, rule in templates.items()
    }
    if nonzero_parameters:
        parameters = {
            "first": {
                "weights": jnp.array([[0.2, -0.4]]),
                "bias": jnp.array([-0.3, 0.5]),
            },
            "second": {
                "weights": jnp.array([[-0.1, 0.6]]),
                "bias": jnp.array([0.4, -0.2]),
            },
        }
    # Every signal has four action branches. Each branch regenerates its reading
    # from that branch's first action; the second policy never sees the signal.
    signal = jnp.repeat((jnp.arange(4096) + 0.5) / 2048 - 1, 4)
    first = jnp.tile(jnp.array([0, 0, 1, 1]), 4096)
    second = jnp.tile(jnp.array([0, 1, 0, 1]), 4096)
    reading = signal + first
    observations = {"first": {"signal": signal}, "second": {"reading": reading}}
    actions = {"first": first, "second": second}
    returns = signal * first + second * (2 * signal + first)

    def branch_probabilities(parameters):
        first_rule = replace(templates["first"], **parameters["first"])
        second_rule = replace(templates["second"], **parameters["second"])
        first_probabilities = jax.vmap(
            lambda x, a: first_rule.probabilities(signal=x)[a]
        )(signal, first)
        second_probabilities = jax.vmap(
            lambda y, a: second_rule.probabilities(reading=y)[a]
        )(reading, second)
        return first_probabilities * second_probabilities

    exact = jax.grad(
        lambda params: jnp.mean(4 * branch_probabilities(params) * returns)
    )(parameters)
    gradients = jax.grad(_score_loss)(
        parameters,
        templates,
        observations,
        actions,
        4 * branch_probabilities(parameters) * returns,
        0.0,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(gradients),
        jax.tree_util.tree_leaves(exact),
        strict=True,
    ):
        assert actual == pytest.approx(-expected, abs=1e-6)
    if not nonzero_parameters:
        assert gradients["first"]["weights"] == pytest.approx(
            jnp.array([[1 / 12, -1 / 12]]), abs=1e-6
        )
        assert gradients["second"]["weights"] == pytest.approx(
            jnp.array([[7 / 24, -7 / 24]]), abs=1e-6
        )
        for decision in gradients.values():
            assert decision["bias"] == pytest.approx(
                jnp.array([1 / 8, -1 / 8]), abs=1e-6
            )

    trajectory = _trajectory(snapshot, templates)
    sampled_returns, sampled_observations, sampled_actions = jax.jit(
        jax.vmap(trajectory, in_axes=(None, 0))
    )(parameters, jax.random.split(jax.random.PRNGKey(8), 65536))
    assert set(sampled_observations["second"]) == {"reading"}
    assert jnp.array_equal(
        sampled_observations["second"]["reading"],
        sampled_observations["first"]["signal"] + sampled_actions["first"],
    )
    sampled_gradients = jax.grad(_score_loss)(
        parameters,
        templates,
        sampled_observations,
        sampled_actions,
        sampled_returns,
        0.5,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(sampled_gradients),
        jax.tree_util.tree_leaves(exact),
        strict=True,
    ):
        assert actual == pytest.approx(-expected, abs=0.015)


@pytest.mark.parametrize("remember_first", [False, True])
def test_solver_learns_joint_policy_and_reuses_local_information(remember_first):
    diagram = _sequential_diagram(remember_first)
    solution = solve(diagram, method="policy_gradient", num_steps=400, num_samples=512)
    assert set(solution.policy) == {"first", "second"}
    for name, rule in solution.policy.items():
        assert isinstance(rule, SoftmaxPolicy)
        assert rule.family is diagram[name].policy_family
        assert rule.parents == diagram[name].parents
        assert rule.states == diagram[name].states
        assert rule.parent_shapes == tuple(() for _ in rule.parents)
    first, second = solution.policy["first"], solution.policy["second"]
    assert isinstance(first, SoftmaxPolicy)
    assert isinstance(second, SoftmaxPolicy)
    assert first.parent_states == (None,)
    assert first.probabilities(signal=jnp.array(-0.8))[0] > 0.95
    assert first.probabilities(signal=jnp.array(0.8))[1] > 0.95
    assert second.parent_states == (
        (None, ("off", "on")) if remember_first else (None,)
    )
    assert second.weights.shape == ((3, 2) if remember_first else (1, 2))
    for reading, action in [(-0.8, 0), (1.8, 1)]:
        information = {"reading": jnp.array(reading)}
        if remember_first:
            information["first"] = jnp.array(action)
        assert second.probabilities(**information)[action] > 0.95
    # The unrestricted optimum chooses first=1 above signal=-1/3, then second=first.
    # Its utility is integral_{-1/3}^1 (3*x + 1)/2 dx = 4/3.
    assert solution.expected_utility == pytest.approx(4 / 3, abs=0.12)
    draws = samples(
        diagram.snapshot(),
        policy=solution.policy,
        query=["signal", "first", "reading", "early_payoff", "late_payoff"],
        num_samples=10000,
        rng_key=jax.random.PRNGKey(22),
    )
    assert jnp.array_equal(draws["reading"], draws["signal"] + draws["first"])
    assert jnp.mean(draws["early_payoff"] + draws["late_payoff"]) == pytest.approx(
        4 / 3, abs=0.1
    )
    payoff = infer(
        diagram,
        ["early_payoff", "late_payoff"],
        policy=solution.policy,
        num_samples=10000,
        rng_key=jax.random.PRNGKey(23),
    )
    assert payoff["early_payoff"].mean() + payoff[
        "late_payoff"
    ].mean() == pytest.approx(solution.expected_utility, abs=0.1)
    assert solution.render(diagram).count("stochastic") == 2
    with pytest.raises(TypeError, match="Unexpected information"):
        second.probabilities(reading=jnp.array(0.0), signal=jnp.array(0.0))


@pytest.mark.parametrize(
    "families",
    [
        (LinearSoftmax, LinearSoftmax, LinearSoftmax),
        (TabularSoftmax, TabularSoftmax, TabularSoftmax),
        (LinearSoftmax, TabularSoftmax, LinearSoftmax),
    ],
)
def test_three_decisions_with_different_action_spaces(families):
    diagram = InfluenceDiagram()
    diagram.add_node(
        DecisionNode("first", states=("off", "on"), policy_family=families[0]())
    )
    diagram.add_node(
        DecisionNode(
            "second",
            parents=("first",),
            states=("low", "medium", "high"),
            policy_family=families[1](),
        )
    )
    diagram.add_node(
        DecisionNode(
            "third",
            parents=("second",),
            states=("off", "on"),
            policy_family=families[2](),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("first", "second", "third"),
            values=lambda first, second, third: first + (second == 2) + third,
        )
    )
    solution = solve(diagram, method="policy_gradient", num_steps=150, num_samples=256)
    for name, shape in [("first", (0, 2)), ("second", (2, 3)), ("third", (3, 2))]:
        rule = solution.policy[name]
        if isinstance(rule, SoftmaxPolicy):
            assert rule.weights.shape == shape
        else:
            assert isinstance(rule, TabularPolicy)
            assert rule.logits.shape == ((2,) if name == "first" else shape)
    assert solution.expected_utility > 2.8


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


@pytest.mark.parametrize("make_diagram", [_signal_diagram, _sequential_diagram])
def test_seed_reproduces_fitted_parameters_and_independent_evaluation(make_diagram):
    diagram = make_diagram()
    first = solve(diagram, method="policy_gradient", num_steps=20, num_samples=64)
    second = solve(diagram, method="policy_gradient", num_steps=20, num_samples=64)
    parameters = {}
    for name, rule in first.policy.items():
        other = second.policy[name]
        assert isinstance(rule, SoftmaxPolicy)
        assert isinstance(other, SoftmaxPolicy)
        assert jnp.array_equal(rule.weights, other.weights)
        assert jnp.array_equal(rule.bias, other.bias)
        parameters[name] = {"weights": rule.weights, "bias": rule.bias}
    assert first.expected_utility == second.expected_utility
    information_key, _, _, evaluation_key = jax.random.split(jax.random.PRNGKey(0), 4)
    templates = _initialize_policy(diagram.snapshot(), information_key)
    returns, _, _ = jax.jit(
        jax.vmap(_trajectory(diagram.snapshot(), templates), in_axes=(None, 0))
    )(parameters, jax.random.split(evaluation_key, 64))
    assert first.expected_utility == float(jnp.mean(returns))


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


def test_solver_requires_family_for_every_decision():
    diagram = _signal_diagram()
    diagram["action"].policy_family = None
    with pytest.raises(ValueError, match="'action' must declare"):
        solve(diagram, method="policy_gradient")
    diagram["action"].policy_family = LinearSoftmax()
    diagram.add_node(DecisionNode("other", states=("off", "on")))
    with pytest.raises(ValueError, match="'other' must declare"):
        solve(diagram, method="policy_gradient")


def test_direct_policy_gradient_solver_requires_a_decision():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Normal(0.0, 1.0)))
    with pytest.raises(ValueError, match="at least one decision"):
        policy_gradient_solve(diagram.snapshot())


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
