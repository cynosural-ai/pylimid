"""Fit influence-diagram policies with joint policy gradients."""

import jax

from pylimid.errors import InferenceError
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.node import NodeKind
from pylimid.solver.policy_gradient import solve as fit_policy
from pylimid.solver.solution import Solution

__all__ = ["solve"]


def solve(
    diagram: InfluenceDiagram,
    *,
    num_samples: int = 2000,
    num_steps: int = 500,
    learning_rate: float = 0.05,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Jointly fit the declared decision policies to maximize expected total utility.

    Every decision must declare `LinearSoftmax` or `TabularSoftmax`. Linear rules
    accept numerical, categorical, or mixed information; tabular rules require scalar
    categorical parents. Each rule receives only its declared information parents,
    including earlier observations and actions that are explicitly remembered.

    Fresh complete scenarios supply score-function gradients for Adam updates. The
    returned policy is the final training iterate, evaluated on independent scenarios.
    Its actions remain stochastic during forward evaluation with `infer`. Training
    has no global or local optimality guarantee, even within the declared families.

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
        InferenceError: If the diagram contains no decisions. Use `infer` for a
            Bayesian network without decisions.
        ValueError: If diagram validation fails, a decision lacks a supported family,
            tabular information is not scalar and categorical, training budgets or
            the learning rate are invalid, a chance distribution conflicts with its
            declared states, or training or evaluation produces non-finite results.
    """
    snapshot = diagram.snapshot()
    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires at least one decision node. "
            "Use infer() for a Bayesian network without decisions."
        )
    return fit_policy(
        snapshot,
        num_samples=num_samples,
        num_steps=num_steps,
        learning_rate=learning_rate,
        rng_key=rng_key,
    )
