"""
Posterior and forward sampling from translated NumPyro models.

Usage::

    from pylimid.numpyro import samples

    posterior = samples(diagram.snapshot(), observed={"wet_grass": 1})

For posterior objects and query validation, use `infer`. `samples` accepts a
validated snapshot and returns raw JAX arrays.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive

from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.node import NodeKind
from pylimid.graph.utility_node import UtilityNode
from pylimid.numpyro.model import _to_model
from pylimid.policy import Policy, _policy_tables

__all__ = ["samples"]


def samples(
    snapshot: DiagramSnapshot,
    *,
    observed: dict[str, Any] | None = None,
    query: list[str] | None = None,
    num_samples: int = 2000,
    num_warmup: int = 500,
    rng_key: Any = None,
    policy: Policy | None = None,
) -> dict[str, jax.Array]:
    """
    Draw samples from a validated snapshot.

    When ``observed`` holds no chance node (it is empty or only binds
    decisions) the draws come directly from the prior joint (independent,
    one pass). With observed chance nodes they come from the posterior:
    exact discrete enumeration when every latent node is
    discrete, NUTS otherwise (``num_warmup`` warmup steps, then
    ``num_samples`` draws; discrete latents are enumerated out of NUTS and
    resampled conditioned on the continuous posterior).

    Args:
        snapshot: A validated, topologically-ordered view of an influence
            diagram. Utility nodes never affect the draws.
        observed: Node-name to observed-value map, holding the observed
            chance nodes and fixed decisions. Empty or ``None`` means
            forward (prior) sampling of a chance-node-only diagram.
        query: Names of the nodes to return. ``None`` (the default) returns
            every chance and decision node; observed nodes appear as their
            clamped value broadcast over the draws. A utility node in
            *query* is evaluated on its parents' values in every draw.
        num_samples: Number of draws.
        num_warmup: NUTS warmup steps; ignored on the forward and
            enumeration paths.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.
        policy: State-dependent decision rules mapping information-set
            assignments to action indices.

    Returns:
        Dict mapping node name to a JAX array with shape
        ``(num_samples, *node_shape)``. Scalar nodes have shape ``(num_samples,)``.

    Raises:
        InferenceError: If a policy names a non-decision node, omits an
            information-set assignment, or contains an invalid action.
        KeyError: If a policy names a node that is not in the snapshot.
        ValueError: If a policy has a continuous information-set parent, or a chance
            distribution conflicts with its ``states``.
    """
    observed = observed or {}
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    policy_tables = _policy_tables(snapshot, policy or {})
    model = _to_model(snapshot, observed, policy_tables)
    draws = _draw(model, snapshot, observed, num_samples, num_warmup, rng_key)
    # Enumerated deterministic sites may be absent or still carry enumeration
    # dimensions. Resolve them from joint posterior draws, parents before children.
    for name, node in snapshot.nodes:
        if name not in policy_tables:
            continue
        table = policy_tables[name]
        if node.parents:
            draws[name] = table[tuple(draws[parent] for parent in node.parents)]
        else:
            draws[name] = jnp.broadcast_to(table, (num_samples,))

    if query is None:
        return draws
    nodes = dict(snapshot.nodes)
    for name in query:
        node = nodes[name]
        if isinstance(node, UtilityNode):
            draws[name] = _evaluate_utility(node, draws, num_samples)
    return {name: draws[name] for name in query}


def _evaluate_utility(
    node: UtilityNode, draws: dict[str, jax.Array], num_samples: int
) -> jax.Array:
    """The utility's value in every draw, computed from its parents' draws."""
    values = node.values
    assert values is not None
    if not node.parents:
        return jnp.broadcast_to(values(), (num_samples,))
    parent_draws = {parent: draws[parent] for parent in node.parents}
    return jax.vmap(lambda parents: values(**parents))(parent_draws)


def _draw(
    model: Callable[[], dict[str, Any]],
    snapshot: DiagramSnapshot,
    observed: dict[str, Any],
    num_samples: int,
    num_warmup: int,
    rng_key: Any,
) -> dict[str, jax.Array]:
    """Run the sampling machinery for ``model`` and return all site draws."""
    if not _has_chance_evidence(snapshot, observed):
        return Predictive(model, num_samples=num_samples)(rng_key)
    if _has_continuous_latent(model, rng_key):
        mcmc = MCMC(
            NUTS(model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            progress_bar=False,
        )
        mcmc.run(rng_key)
        posterior = mcmc.get_samples()
        # NUTS marginalises the enumerated discrete sites out of ``posterior``;
        # this pass resamples them conditioned on the continuous draws. The
        # combined dict covers every node: continuous latents from
        # ``posterior``, discrete latents and observed sites from here.
        conditional = Predictive(
            model, posterior_samples=posterior, infer_discrete=True
        )(rng_key)
        return {**posterior, **conditional}
    return Predictive(model, num_samples=num_samples, infer_discrete=True)(rng_key)


def _has_chance_evidence(snapshot: DiagramSnapshot, observed: dict[str, Any]) -> bool:
    """Whether any chance node is observed (bound decisions don't count)."""
    return any(
        name in observed and node.kind is NodeKind.CHANCE
        for name, node in snapshot.nodes
    )


def _has_continuous_latent(model: Callable[[], dict[str, Any]], rng_key: Any) -> bool:
    """Read continuous latent support from distributions in a seeded execution."""
    trace = handlers.trace(handlers.seed(model, rng_key)).get_trace()
    return any(
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].support.is_discrete
        for site in trace.values()
    )
