"""
NumPyro inference backend — forward sampling, MCMC, and SVI.

Thin wrappers around :func:`~decisionpy.inference._numpyro_model.to_model`.
Import this module to use NumPyro-based inference; requires the ``numpyro``
extra (``pip install decisionpy[numpyro]``).

Usage::

    from decisionpy.inference.numpyro import forward_sample, mcmc

    samples = forward_sample(diagram.snapshot(), num_samples=1000, rng_key=key)
    posterior = mcmc(diagram.snapshot(), observed={"wet_grass": 1}, ...)

API design note
---------------
This module exposes engine-specific functions.  The unified ``infer()``
entry-point in :mod:`decisionpy.inference` dispatches here automatically,
so most users should call ``infer()`` rather than import this directly.
"""

from __future__ import annotations

from typing import Any

from decisionpy.graph.diagram import Snapshot
from decisionpy.inference._numpyro_model import to_model

__all__ = ["forward_sample", "to_model"]


def forward_sample(
    snapshot: Snapshot,
    num_samples: int = 1000,
    rng_key: Any = None,
) -> dict[str, Any]:
    """
    Draw *num_samples* from the prior joint distribution.

    Equivalent to ``Predictive(to_model(snapshot), num_samples)(rng_key)``.

    :returns: Dict mapping node name to a 1-D JAX array of samples.
    """
    import jax
    from numpyro.infer import Predictive

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    model = to_model(snapshot)
    return Predictive(model, num_samples=num_samples)(rng_key)
