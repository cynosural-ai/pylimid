"""
Internal forward and posterior sampling for validated diagram snapshots.

The inference API prepares relevant nodes and bindings before requesting raw JAX
arrays from this layer, then wraps the requested draws in posterior result objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive

from pylimid._numpyro.distributions import supports_count_enumeration
from pylimid._numpyro.model import build_model
from pylimid.errors import InferenceError
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.nodes import DecisionNode, NodeKind, UtilityNode
from pylimid.policy import CategoricalPolicy, DecisionRule
from pylimid.policy._binding import bind_policy_rules

__all__ = ["samples"]

#: Joint draws keyed by node name. Each array has shape
#: ``(num_samples, *node_shape)``; scalar nodes have shape ``(num_samples,)``.
#: The same sample index across nodes identifies one joint draw, so relationships
#: between variables are preserved. Categories contain integer state indices;
#: numerical nodes and utilities contain their actual values.
type SampleBatch = dict[str, jax.Array]


def samples(
    snapshot: DiagramSnapshot,
    *,
    observed: dict[str, Any] | None = None,
    query: list[str] | None = None,
    num_samples: int = 2000,
    num_warmup: int = 500,
    rng_key: Any = None,
    policy: Mapping[str, DecisionRule] | None = None,
) -> SampleBatch:
    """
    Draw samples from a validated snapshot.

    When ``observed`` holds no chance node (it is empty or only binds
    decisions) the draws come directly from the prior joint (independent,
    one pass). With observed chance nodes they come from the posterior:
    exact discrete enumeration when every latent node is categorical or a supported
    finite count, NUTS otherwise (``num_warmup`` warmup steps, then
    ``num_samples`` draws; discrete latents are enumerated out of NUTS and
    resampled conditioned on the continuous posterior). Numerical counts can be
    forward sampled or observed as likelihoods. Count latents require scalar
    distributions with NumPyro enumeration over a fixed integer range starting at
    zero. For Binomial, use a Python integer ``total_count``. Non-static bounds,
    including parent-dependent trial counts, retain NumPyro's or JAX's errors.

    Args:
        snapshot: A validated, topologically-ordered view of an influence
            diagram. Utility nodes never affect the draws.
        observed: Node-name to observed-value map, holding the observed
            chance nodes and fixed decisions. Empty or ``None`` means forward
            sampling, using ``policy`` for any decisions.
        query: Names of the nodes to return. ``None`` (the default) returns
            every chance and decision node; observed nodes appear as their
            clamped value broadcast over the draws. A utility node in
            *query* is evaluated on its parents' values in every draw.
        num_samples: Number of draws.
        num_warmup: NUTS warmup steps; ignored on the forward and
            enumeration paths.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.
        policy: Decision names mapped to complete action tables, JAX-traceable
            deterministic functions, or fitted `CategoricalPolicy` rules. Functions
            receive only declared information parents as keyword arguments and return
            a scalar integer action index. Fitted rules sample actions from learned
            probabilities. Functions and fitted rules support forward evaluation only.

    Returns:
        Dict mapping node name to a JAX array with shape
        ``(num_samples, *node_shape)``. Scalar nodes have shape ``(num_samples,)``.

    Raises:
        InferenceError: If a policy names a non-decision node, omits an
            information-set assignment, or contains an invalid action. Also raised
            when a numerical count latent lacks supported enumeration or has a
            non-scalar batch shape.
        KeyError: If a policy names a node that is not in the snapshot.
        ValueError: If a policy has an information-set parent without named states, or
            a chance distribution conflicts with its ``states``.
        NotImplementedError: If function or fitted stochastic policies use chance-node
            evidence.
    """
    observed = observed or {}
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    policy = policy or {}
    policy_rules = bind_policy_rules(snapshot, policy, observed)
    model = build_model(snapshot, observed, policy_rules)
    draws = _draw(model, snapshot, observed, num_samples, num_warmup, rng_key)
    # Enumerated deterministic sites may be absent or still carry enumeration
    # dimensions. Resolve them from joint posterior draws, parents before children.
    for name, node in snapshot.nodes:
        if name not in policy_rules:
            continue
        assert isinstance(node, DecisionNode)
        rule = policy_rules[name]
        if isinstance(rule, CategoricalPolicy):
            continue
        if node.parents:
            draws[name] = jax.vmap(rule)(
                **{parent: draws[parent] for parent in node.parents}
            )
        else:
            draws[name] = jnp.broadcast_to(rule(), (num_samples,))
        if callable(policy[name]):
            assert node.states is not None
            if jnp.any((draws[name] < 0) | (draws[name] >= len(node.states))):
                raise InferenceError(f"Policy for {name!r} returned an invalid action.")

    if query is None:
        return draws
    nodes = dict(snapshot.nodes)
    for name in query:
        node = nodes[name]
        if isinstance(node, UtilityNode):
            draws[name] = _evaluate_utility(node, draws, num_samples)
    return {name: draws[name] for name in query}


def _evaluate_utility(
    node: UtilityNode, draws: SampleBatch, num_samples: int
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
) -> SampleBatch:
    """Run the sampling machinery for ``model`` and return all site draws."""
    if not _has_chance_evidence(snapshot, observed):
        return Predictive(model, num_samples=num_samples)(rng_key)
    if _validate_posterior_latents(snapshot, model, rng_key):
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


def _validate_posterior_latents(
    snapshot: DiagramSnapshot, model: Callable[[], dict[str, Any]], rng_key: Any
) -> bool:
    """Check latent enumeration support and report whether continuous latents remain."""
    trace = handlers.trace(handlers.seed(model, rng_key)).get_trace()
    nodes = dict(snapshot.nodes)
    has_continuous = False
    for name, site in trace.items():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        fn = site["fn"]
        if not fn.support.is_discrete:
            has_continuous = True
        elif not nodes[name].is_categorical:
            if fn.batch_shape:
                raise InferenceError(
                    f"Posterior inference for count node {name!r} requires a scalar "
                    f"distribution; its batch shape is {fn.batch_shape}. Batched "
                    "numerical counts support forward sampling and observations only."
                )
            if not supports_count_enumeration(fn):
                raise InferenceError(
                    "Posterior inference with unobserved numerical discrete nodes "
                    "requires NumPyro enumeration of a fixed zero-based integer "
                    f"range: {name!r} uses {type(fn).__name__}. Observe this node "
                    "or use forward sampling without chance-node evidence."
                )
    return has_continuous
