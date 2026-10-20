"""Fit one declared linear-softmax decision by expected-utility policy gradients."""

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
from pylimid.policy_family import LinearSoftmax, SoftmaxPolicy
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
    Fit a single decision's declared `LinearSoftmax` family with Adam.

    Each update samples fresh complete scenarios and estimates a score-function
    gradient of expected total utility. The previous batch's mean utility supplies
    an action-independent baseline. Observations, sampled actions, utilities, and
    the baseline are held fixed when differentiating policy log probabilities.

    The final stochastic policy is evaluated on independent fresh scenarios. Its
    expected utility is an estimate, with no global or local optimality guarantee.
    Multiple decisions and other policy families are not supported by this solver.

    Args:
        snapshot: A validated diagram with exactly one decision declaring
            ``policy_family=LinearSoftmax()``.
        num_samples: Scenarios per training update and in the final evaluation.
        num_steps: Number of optimizer updates.
        learning_rate: Positive, finite Adam step size.
        rng_key: JAX random key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` containing the fitted `SoftmaxPolicy` and an independent
        Monte Carlo estimate of its expected total utility.

    Raises:
        NotImplementedError: If the diagram does not have exactly one decision.
        ValueError: If the decision has no declared `LinearSoftmax` family, a budget
            is non-positive, the learning rate is invalid, or optimization produces
            non-finite parameters or expected utility. Also raised if a chance
            distribution conflicts with its ``states``.
    """
    decisions = [node for _, node in snapshot.nodes if isinstance(node, DecisionNode)]
    if len(decisions) != 1:
        raise NotImplementedError(
            "Policy-gradient solving currently supports exactly one decision."
        )
    decision = decisions[0]
    if not isinstance(decision.policy_family, LinearSoftmax):
        raise ValueError(
            f"Decision {decision.name!r} must declare policy_family=LinearSoftmax()."
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
    nodes = {
        name: node for name, node in snapshot.nodes if isinstance(node, ChanceNode)
    }
    parent_states = tuple(nodes[parent].states for parent in decision.parents)
    assert decision.states is not None
    information = _sample_information(snapshot, decision, information_key)
    feature_count = decision.policy_family._features(
        decision.parents, parent_states, information
    ).size
    template = SoftmaxPolicy(
        family=decision.policy_family,
        parents=decision.parents,
        parent_states=parent_states,
        parent_shapes=tuple(
            jnp.shape(information[parent]) for parent in decision.parents
        ),
        states=decision.states,
        weights=jnp.zeros((feature_count, len(decision.states))),
        bias=jnp.zeros(len(decision.states)),
    )
    parameters = {"weights": template.weights, "bias": template.bias}
    trajectory = _trajectory(snapshot, decision, template)
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
                parameters, template, observations, actions, returns, baseline
            )
            return (update(step, gradients, state), key, jnp.mean(returns)), None

        (state, _, _), _ = jax.lax.scan(
            iteration, (initialize(parameters), key, baseline), jnp.arange(num_steps)
        )
        return get_parameters(state)

    fitted_parameters = train(parameters, training_key, baseline)
    if not all(
        bool(jnp.all(jnp.isfinite(value))) for value in fitted_parameters.values()
    ):
        raise ValueError("Policy optimization produced non-finite parameters.")
    fitted = replace(template, **fitted_parameters)
    returns, _, _ = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        fitted_parameters, jax.random.split(evaluation_key, num_samples)
    )
    expected_utility = float(jnp.mean(returns))
    if not isfinite(expected_utility):
        raise ValueError("Policy evaluation produced non-finite expected utility.")
    return Solution(
        policy={decision.name: fitted},
        expected_utility=expected_utility,
        method="policy_gradient",
    )


def _sample_information(snapshot, decision, key):
    """Sample actual parent values before initializing the policy's feature shape."""
    values = {}
    keys = iter(jax.random.split(key, len(snapshot.nodes)))
    for name, node in snapshot.nodes:
        if name == decision.name:
            break
        if isinstance(node, ChanceNode):
            assert node.dist is not None
            distribution = node.dist(
                **{parent: values[parent] for parent in node.parents}
            )
            _validate_chance_distribution(node, distribution)
            values[name] = distribution.sample(next(keys))
    return {parent: values[parent] for parent in decision.parents}


def _score_loss(parameters, template, observations, actions, returns, baseline):
    """Negative score-function surrogate, with all sampled quantities held fixed."""
    observations, actions, returns, baseline = jax.tree_util.tree_map(
        jax.lax.stop_gradient, (observations, actions, returns, baseline)
    )
    rule = replace(template, **parameters)
    log_probabilities = jax.vmap(lambda row, action: rule.log_prob(action, **row))(
        observations, actions
    )
    return -jnp.mean((returns - baseline) * log_probabilities)


def _trajectory(snapshot, decision, template):
    """Simulate one scenario under the current stochastic rule."""
    utility_nodes = [
        node for _, node in snapshot.nodes if isinstance(node, UtilityNode)
    ]
    num_random = sum(isinstance(node, ChanceNode) for _, node in snapshot.nodes) + 1

    def simulate(parameters, key):
        rule = replace(template, **parameters)
        keys = iter(jax.random.split(key, num_random))
        values = {}
        for name, node in snapshot.nodes:
            if isinstance(node, UtilityNode):
                continue
            parents = {parent: values[parent] for parent in node.parents}
            if isinstance(node, DecisionNode):
                values[name] = rule.sample(next(keys), **parents)
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
            {parent: values[parent] for parent in decision.parents},
            values[decision.name],
        )

    return simulate
