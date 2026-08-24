"""
NumPyro inference backend — posterior and forward sampling.

Thin wrappers around to_model().
Import this module to use NumPyro-based inference; requires ``numpyro``, a
declared dependency of decisionpy.

Usage::

    from decisionpy.inference.numpyro import samples

    posterior = samples(diagram.snapshot(), observed={"wet_grass": 1})

API design note
---------------
This module exposes engine-specific functions.  The unified ``infer()``
entry-point in decisionpy.inference dispatches here automatically,
so most users should call ``infer()`` rather than import this directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from numpyro.infer import MCMC, NUTS, Predictive

from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.node import NodeKind
from decisionpy.inference.numpyro.model import to_model

__all__ = ["samples", "to_model"]


def samples(
    snapshot: Snapshot,
    *,
    observed: dict[str, Any] | None = None,
    query: list[str] | None = None,
    num_samples: int = 2000,
    num_warmup: int = 500,
    rng_key: Any = None,
) -> dict[str, jax.Array]:
    """
    Draw samples from a validated snapshot.

    With ``observed`` empty or ``None`` the draws come directly from the
    prior joint (independent, one pass). With observations they come from
    the posterior: exact discrete enumeration when every latent node is
    discrete, NUTS otherwise (``num_warmup`` warmup steps, then
    ``num_samples`` draws; discrete latents are enumerated out of NUTS and
    resampled conditioned on the continuous posterior).

    Args:
        snapshot: A validated, topologically-ordered view of an influence
            diagram. Every decision must be bound in *observed*; utility
            nodes are ignored.
        observed: Node-name to observed-value map. Empty or ``None`` means
            forward (prior) sampling of a chance-node-only diagram.
        query: Names of the nodes to return. ``None`` (the default) returns
            every node; observed nodes appear as their clamped value
            broadcast over the draws.
        num_samples: Number of draws.
        num_warmup: NUTS warmup steps; ignored on the forward and
            enumeration paths.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        Dict mapping node name to a 1-D JAX array of ``num_samples`` draws.
    """
    observed = observed or {}
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    model = to_model(snapshot, observed=observed)
    draws = _draw(model, snapshot, observed, num_samples, num_warmup, rng_key)

    if query is None:
        return draws
    return {name: draws[name] for name in query}


def _draw(
    model: Callable[[], dict[str, Any]],
    snapshot: Snapshot,
    observed: dict[str, Any],
    num_samples: int,
    num_warmup: int,
    rng_key: Any,
) -> dict[str, jax.Array]:
    """Run the sampling machinery for ``model`` and return all site draws."""
    if not observed:
        return Predictive(model, num_samples=num_samples)(rng_key)
    if _has_continuous_latent(snapshot, observed):
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


def _has_continuous_latent(snapshot: Snapshot, observed: dict[str, Any]) -> bool:
    """Whether any unobserved chance node is continuous."""
    for name, node in snapshot.nodes:
        if name in observed:
            continue
        if node.kind is NodeKind.CHANCE and not node.is_discrete:
            return True
    return False
