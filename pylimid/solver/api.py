"""Joint expected-utility optimization with score-function policy gradients."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam

from pylimid._numpyro.distributions import validate_chance_distribution
from pylimid.errors import SolveError
from pylimid.graph.diagram import DiagramSnapshot, InfluenceDiagram
from pylimid.graph.nodes import ChanceNode, DecisionNode, UtilityNode
from pylimid.policy import (
    CategoricalPolicy,
    ContinuousPolicy,
    DecisionInformation,
    FittedPolicy,
    PolicyParameters,
)
from pylimid.solver.solution import Solution

__all__ = ["solve"]


class _Trajectory(NamedTuple):
    """
    Total utility, local decision information, and actions from one scenario.

    Applying ``jax.vmap`` produces the same record with a leading scenario axis on
    every array. Matching positions across its fields refer to the same scenario.

    Attributes:
        utility: Scalar sum of all utility-node values. A vector after batching.
        information: Decision names mapped to their declared information-parent
            values. Each value retains its node shape before batching.
        actions: Decision names mapped to scalar categorical indices or real actions.
    """

    utility: float | jax.Array
    information: dict[str, DecisionInformation]
    actions: dict[str, jax.Array]


def solve(
    diagram: InfluenceDiagram,
    *,
    num_samples: int = 10000,
    num_steps: int = 500,
    learning_rate: float = 0.05,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Jointly fit the declared decision policies to maximize expected total utility.

    Every decision must declare a `PolicyFamily`. `LinearSoftmax` accepts numerical,
    categorical, or mixed information; `TabularSoftmax` requires scalar categorical
    parents. `DistributionPolicy` composes a continuous distribution with a
    `ParameterModel`, such as `Linear`, for scalar `Real` or `Interval` actions.
    Each rule receives only its declared information parents, including
    earlier observations and actions that are explicitly remembered.

    Fresh complete scenarios supply score-function gradients for Adam updates. The
    returned policy is the final training iterate, evaluated on independent scenarios.
    Pass the policy to `infer` for forward evaluation or conditioning on chance-node
    evidence. Its action probabilities or densities contribute to the joint model.
    Training has no global or local optimality guarantee, even within the declared
    families.

    Chance distributions and utility functions must be JAX-traceable. Use ``jnp``
    operations rather than Python conditionals or casts on sampled values.

    Args:
        diagram: Influence diagram with at least one decision. Structural and node
            configuration validation runs before training.
        num_samples: Scenarios per training update and in the independent evaluation.
        num_steps: Number of optimizer updates.
        learning_rate: Positive, finite Adam step size.
        rng_key: JAX random key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` containing a fitted stochastic rule per decision and an
        independent Monte Carlo estimate of their expected total utility.

    Raises:
        ValueError: If diagram validation fails, the diagram contains no decisions,
            a decision lacks a family, tabular information is not scalar and
            categorical, training budgets or the learning rate are invalid, or a
            chance distribution conflicts with its declared states, or an action
            distribution is incompatible with its domain. Use `infer` for a Bayesian
            network without decisions.
        SolveError: If optimization produces non-finite parameters or final
            evaluation produces non-finite expected utility.
        NotImplementedError: If a continuous action distribution has unsupported
            support metadata.
    """
    snapshot = diagram.snapshot()
    decisions = [node for _, node in snapshot.nodes if isinstance(node, DecisionNode)]
    if not decisions:
        raise ValueError(
            "solve() requires at least one decision node. "
            "Use infer() for a Bayesian network without decisions."
        )
    for decision in decisions:
        if decision.policy_family is None:
            raise ValueError(
                f"Decision {decision.name!r} must declare "
                "a policy_family implementing PolicyFamily."
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
    parameters = {name: rule.parameters for name, rule in templates.items()}
    trajectory = _trajectory(snapshot, templates)
    pilot = trajectory(parameters, shape_key)
    baseline = jnp.zeros_like(jnp.mean(pilot.utility))
    initialize, update, get_parameters = adam(learning_rate)

    @jax.jit
    def train(parameters, key, baseline):
        def iteration(carry, step):
            state, key, baseline = carry
            parameters = get_parameters(state)
            key, batch_key = jax.random.split(key)
            batch = jax.vmap(trajectory, in_axes=(None, 0))(
                parameters, jax.random.split(batch_key, num_samples)
            )
            gradients = jax.grad(_score_loss)(
                parameters,
                templates,
                batch.information,
                batch.actions,
                batch.utility,
                baseline,
            )
            return (update(step, gradients, state), key, jnp.mean(batch.utility)), None

        (state, _, _), _ = jax.lax.scan(
            iteration, (initialize(parameters), key, baseline), jnp.arange(num_steps)
        )
        return get_parameters(state)

    fitted_parameters = train(parameters, training_key, baseline)
    if not all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(fitted_parameters)
    ):
        raise SolveError("Policy optimization produced non-finite parameters.")
    evaluation = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))(
        fitted_parameters, jax.random.split(evaluation_key, num_samples)
    )
    expected_utility = float(jnp.mean(evaluation.utility))
    if not isfinite(expected_utility):
        raise SolveError("Policy evaluation produced non-finite expected utility.")
    return Solution(
        policy={
            name: replace(rule, parameters=fitted_parameters[name])
            for name, rule in templates.items()
        },
        expected_utility=expected_utility,
    )


def _initialize_policy(snapshot: DiagramSnapshot, key: jax.Array) -> FittedPolicy:
    """Bind information shapes during a walk under the initialized local policies."""
    nodes = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    templates: FittedPolicy = {}
    values: dict[str, jax.Array] = {}
    keys = iter(jax.random.split(key, len(nodes)))
    for index, (name, node) in enumerate(nodes.items()):
        parents = {parent: values[parent] for parent in node.parents}
        if isinstance(node, ChanceNode):
            assert node.dist is not None
            distribution = node.dist(**parents)
            validate_chance_distribution(node, distribution)
            values[name] = distribution.sample(next(keys))
        else:
            family = node.policy_family
            assert family is not None
            parent_states = tuple(nodes[parent].states for parent in node.parents)
            parameters = family.initialize(
                jax.random.fold_in(key, index),
                num_actions=None if node.states is None else len(node.states),
                observations=parents,
                parent_states=dict(zip(node.parents, parent_states, strict=True)),
            )
            shapes = tuple(jnp.shape(parents[parent]) for parent in node.parents)
            rule: CategoricalPolicy | ContinuousPolicy
            if node.action_space is not None:
                rule = ContinuousPolicy(
                    family,
                    node.parents,
                    parent_states,
                    shapes,
                    node.action_space,
                    parameters,
                )
            else:
                assert node.states is not None
                rule = CategoricalPolicy(
                    family, node.parents, parent_states, shapes, node.states, parameters
                )
            templates[name] = rule
            values[name] = rule.sample(next(keys), **parents)
    return templates


def _score_loss(
    parameters: dict[str, PolicyParameters],
    templates: FittedPolicy,
    observations: dict[str, Mapping[str, jax.Array]],
    actions: dict[str, jax.Array],
    returns: jax.Array,
    baseline: float | jax.Array,
) -> jax.Array:
    """
    Compute the negative score-function surrogate from a batch of scenarios.

    Args:
        parameters: Trainable parameter dictionaries keyed by decision name.
        templates: Local stochastic rules whose parameters are replaced for scoring.
        observations: Decision names mapped to their information-parent sample arrays.
            Every array has a leading scenario axis followed by its node shape.
        actions: Sampled scalar action vectors keyed by decision name.
        returns: Total utility for each scenario, aligned with observations and actions.
        baseline: Scalar utility baseline, independent of the current sampled actions.

    Returns:
        The scalar surrogate loss, holding sampled quantities and the baseline fixed
        while differentiating the policy log probabilities.
    """
    observations, actions, returns, baseline = jax.tree_util.tree_map(
        jax.lax.stop_gradient, (observations, actions, returns, baseline)
    )
    log_probabilities = 0.0
    for name, template in templates.items():
        rule = replace(template, parameters=parameters[name])
        log_probabilities = log_probabilities + jax.vmap(rule.log_prob)(
            actions[name], **observations[name]
        )
    return -jnp.mean((returns - baseline) * log_probabilities)


def _trajectory(
    snapshot: DiagramSnapshot, templates: FittedPolicy
) -> Callable[[dict[str, PolicyParameters], jax.Array], _Trajectory]:
    """
    Build a simulator for complete scenarios under local stochastic rules.

    Args:
        snapshot: Validated diagram in topological order.
        templates: Initialized stochastic rules with bound information metadata.

    Returns:
        A function of per-decision parameters and a random key, returning a
        `_Trajectory` with total utility, local information, and sampled actions.
    """
    utility_nodes = [
        node for _, node in snapshot.nodes if isinstance(node, UtilityNode)
    ]
    num_random = sum(
        isinstance(node, (ChanceNode, DecisionNode)) for _, node in snapshot.nodes
    )

    def simulate(
        parameters: dict[str, PolicyParameters], key: jax.Array
    ) -> _Trajectory:
        rules = {
            name: replace(rule, parameters=parameters[name])
            for name, rule in templates.items()
        }
        keys = iter(jax.random.split(key, num_random))
        values: dict[str, jax.Array] = {}
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
                validate_chance_distribution(node, distribution)
                values[name] = distribution.sample(next(keys))
        total = 0.0
        for node in utility_nodes:
            assert node.values is not None
            total = total + node.values(
                **{parent: values[parent] for parent in node.parents}
            )
        return _Trajectory(
            utility=total,
            information={
                name: {parent: values[parent] for parent in rule.parents}
                for name, rule in templates.items()
            },
            actions={name: values[name] for name in templates},
        )

    return simulate
