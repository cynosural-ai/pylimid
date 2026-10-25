"""Fit declared softmax decisions by joint expected-utility policy gradients."""

from dataclasses import replace
from math import isfinite

import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam

from pylimid._distributions import _validate_chance_distribution
from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.utility_node import UtilityNode
from pylimid.policy_family import LinearSoftmax, TabularSoftmax
from pylimid.solver.solution import Solution

__all__ = ["solve"]


def solve(
    snapshot: DiagramSnapshot,
    *,
    num_samples: int = 2000,
    num_steps: int = 500,
    learning_rate: float = 0.05,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Fit every decision's declared softmax family jointly with Adam.

    Each update samples fresh complete scenarios and estimates a score-function
    gradient of expected total utility for all decisions. Each rule uses only its
    declared information parents, including any explicitly remembered decisions.
    Earlier actions affect later observations through the simulated chance model.
    The previous batch's mean utility supplies an action-independent baseline.
    Observations, sampled actions, utilities, and the baseline are held fixed when
    differentiating policy log probabilities.

    The final stochastic policy is evaluated on independent fresh scenarios. Its
    expected utility is an estimate, with no global or local optimality guarantee.
    Every decision must declare `LinearSoftmax` or `TabularSoftmax`. Families can be
    mixed in one diagram; tabular rules require scalar categorical information.

    Args:
        snapshot: A validated diagram with at least one decision, each declaring
            a `LinearSoftmax` or `TabularSoftmax` policy family.
        num_samples: Scenarios per training update and in the final evaluation.
        num_steps: Number of optimizer updates.
        learning_rate: Positive, finite Adam step size.
        rng_key: JAX random key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` containing a fitted `SoftmaxPolicy` or `TabularPolicy` per
        decision and an independent Monte Carlo estimate of their joint expected
        total utility.

    Raises:
        ValueError: If there are no decisions, any decision lacks a declared
            supported family, tabular information is not scalar and categorical,
            a budget is non-positive, the learning rate is
            invalid, or optimization produces non-finite parameters or expected
            utility. Also raised if a chance distribution conflicts with its ``states``.
    """
    decisions = [node for _, node in snapshot.nodes if isinstance(node, DecisionNode)]
    if not decisions:
        raise ValueError("Policy-gradient solving requires at least one decision.")
    for decision in decisions:
        if not isinstance(decision.policy_family, (LinearSoftmax, TabularSoftmax)):
            raise ValueError(
                f"Decision {decision.name!r} must declare "
                "policy_family=LinearSoftmax() or TabularSoftmax()."
            )
    if num_samples <= 0 or num_steps <= 0:
        raise ValueError("num_samples and num_steps must be positive.")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    information_key, shape_key, training_key, evaluation_key = jax.random.split(
        rng_key, 4
    )
    templates = _initialize_policy(snapshot, information_key)
    parameters = {name: rule._parameters() for name, rule in templates.items()}
    trajectory = _trajectory(snapshot, templates)
    pilot_utility, _, _ = trajectory(parameters, shape_key)
    baseline = jnp.zeros_like(jnp.mean(pilot_utility))
    initialize, update, get_parameters = adam(learning_rate)

    @jax.jit
    def train(parameters, key, baseline):
        def iteration(carry, step):
            state, key, baseline = carry
            parameters = get_parameters(state)
            key, batch_key = jax.random.split(key)
            returns, observations, actions = jax.vmap(trajectory, in_axes=(None, 0))(
                parameters, jax.random.split(batch_key, num_samples)
            )
            gradients = jax.grad(_score_loss)(
                parameters, templates, observations, actions, returns, baseline
            )
            return (update(step, gradients, state), key, jnp.mean(returns)), None

        (state, _, _), _ = jax.lax.scan(
            iteration, (initialize(parameters), key, baseline), jnp.arange(num_steps)
        )
        return get_parameters(state)

    fitted_parameters = train(parameters, training_key, baseline)
    if not all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(fitted_parameters)
    ):
        raise ValueError("Policy optimization produced non-finite parameters.")
    returns, _, _ = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        fitted_parameters, jax.random.split(evaluation_key, num_samples)
    )
    expected_utility = float(jnp.mean(returns))
    if not isfinite(expected_utility):
        raise ValueError("Policy evaluation produced non-finite expected utility.")
    return Solution(
        policy={
            name: replace(rule, **fitted_parameters[name])
            for name, rule in templates.items()
        },
        expected_utility=expected_utility,
    )


def _initialize_policy(snapshot, key):
    """Bind each rule's feature shape during a walk under uniform initial policies."""
    nodes = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    templates = {}
    values = {}
    keys = iter(jax.random.split(key, len(nodes)))
    for name, node in nodes.items():
        parents = {parent: values[parent] for parent in node.parents}
        if isinstance(node, ChanceNode):
            assert node.dist is not None
            distribution = node.dist(**parents)
            _validate_chance_distribution(node, distribution)
            values[name] = distribution.sample(next(keys))
        else:
            assert node.states is not None
            assert isinstance(node.policy_family, (LinearSoftmax, TabularSoftmax))
            parent_states = tuple(nodes[parent].states for parent in node.parents)
            rule = node.policy_family._initialize(
                node.parents, parent_states, parents, node.states
            )
            templates[name] = rule
            values[name] = rule.sample(next(keys), **parents)
    return templates


def _score_loss(parameters, templates, observations, actions, returns, baseline):
    """Negative score-function surrogate, with all sampled quantities held fixed."""
    observations, actions, returns, baseline = jax.tree_util.tree_map(
        jax.lax.stop_gradient, (observations, actions, returns, baseline)
    )
    log_probabilities = 0.0
    for name, template in templates.items():
        rule = replace(template, **parameters[name])
        log_probabilities = log_probabilities + jax.vmap(rule.log_prob)(
            actions[name], **observations[name]
        )
    return -jnp.mean((returns - baseline) * log_probabilities)


def _trajectory(snapshot, templates):
    """Simulate one complete scenario under all current local stochastic rules."""
    utility_nodes = [
        node for _, node in snapshot.nodes if isinstance(node, UtilityNode)
    ]
    num_random = sum(
        isinstance(node, (ChanceNode, DecisionNode)) for _, node in snapshot.nodes
    )

    def simulate(parameters, key):
        rules = {
            name: replace(rule, **parameters[name]) for name, rule in templates.items()
        }
        keys = iter(jax.random.split(key, num_random))
        values = {}
        for name, node in snapshot.nodes:
            if isinstance(node, UtilityNode):
                continue
            parents = {parent: values[parent] for parent in node.parents}
            if isinstance(node, DecisionNode):
                values[name] = rules[name].sample(next(keys), **parents)
            else:
                assert isinstance(node, ChanceNode)
                assert node.dist is not None
                distribution = node.dist(**parents)
                _validate_chance_distribution(node, distribution)
                values[name] = distribution.sample(next(keys))
        total = 0.0
        for node in utility_nodes:
            assert node.values is not None
            total = total + node.values(
                **{parent: values[parent] for parent in node.parents}
            )
        return (
            total,
            {
                name: {parent: values[parent] for parent in rule.parents}
                for name, rule in templates.items()
            },
            {name: values[name] for name in templates},
        )

    return simulate
