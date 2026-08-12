"""
Unified inference entry-point.

::

    from decisionpy.inference import infer

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": [0.456, 0.544]}  (exact posteriors for discrete BNs)
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, Predictive

from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import NodeKind
from decisionpy.inference.numpyro.model import to_model
from decisionpy.inference.ve import query as ve_query

__all__ = ["InferenceError", "InferenceResult", "infer"]


class InferenceError(Exception):
    """Asked for an inference method that cannot handle the given diagram."""


#: Result of :func:`infer`: one entry per query variable. For a discrete
#: variable (declared ``states``) the value is its posterior probability vector
#: ``[P(0), P(1), ...]`` (summing to 1); for a continuous variable it is a list
#: of posterior samples (MCMC draws).
InferenceResult: TypeAlias = dict[str, list[float]]


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int] | None = None,
    engine: str = "auto",
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Args:
        diagram: A validated influence diagram (chance nodes only for v0).
        query: Names of variables whose posteriors are requested.
        observed: Map from node name to its observed integer state.
        engine: ``"auto"`` (default), ``"ve"``, or ``"numpyro"``.

    Returns:
        ``{var_name: values}`` — one entry per query variable. For a discrete
        variable (declared ``states``) the value is its posterior probability
        vector ``[P(0), P(1), ...]`` (summing to 1); for a continuous variable
        it is a list of posterior samples (MCMC draws).

    Raises:
        InferenceError: If the chosen engine cannot handle the diagram.
    """
    observed = observed or {}
    snapshot = diagram.snapshot()

    if engine == "auto":
        engine = _choose_engine(snapshot)

    if engine == "ve":
        return _infer_ve(snapshot, query, observed)

    if engine == "numpyro":
        return _infer_numpyro(snapshot, query, observed)

    raise InferenceError(
        f"Unknown engine {engine!r}. Choose 'auto', 've', or 'numpyro'."
    )


# -- engine selection ---------------------------------------------------------


def _choose_engine(snapshot) -> str:
    # Node classification is by ``kind``, never by ``is_discrete`` (vacuous
    # for utility nodes). A diagram is a Bayesian network iff every node is a
    # chance node; only then does ``is_discrete`` pick the exact engine.
    nodes = [node for _, node in snapshot.nodes]
    is_bn = all(node.kind is NodeKind.CHANCE for node in nodes)
    if is_bn and all(node.is_discrete for node in nodes):
        return "ve"
    return "numpyro"


# -- engine backends ----------------------------------------------------------


def _infer_ve(snapshot, query, observed) -> InferenceResult:
    try:
        return ve_query(snapshot, variables=query, observed=observed)
    except TypeError as e:
        raise InferenceError(
            f"Variable elimination requires an all-discrete diagram. {e}"
        ) from e


def _infer_numpyro(snapshot, query, observed) -> InferenceResult:
    # The vectorized discrete path applies only to a pure discrete BN: a node
    # counts as discrete engine input only if it is a chance node (utility
    # nodes have no domain — classify by kind, not ``is_discrete``).
    all_discrete = all(
        node.kind is NodeKind.CHANCE and node.is_discrete for _, node in snapshot.nodes
    )
    model = to_model(snapshot, observed=observed)

    if all_discrete:
        return _infer_numpyro_discrete(model, snapshot, query)
    else:
        mcmc = MCMC(
            NUTS(model),
            num_warmup=500,
            num_samples=2000,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0))
        posterior = mcmc.get_samples()
        result = {
            name: [float(x) for x in posterior[name]]
            for name in query
            if name in posterior
        }
        discrete = [name for name in query if name not in result]
        if discrete:
            # Discrete marginals condition on the continuous posterior: hand
            # the MCMC draws to Predictive so funsor samples the enumerated
            # sites given them. A bare ``infer_discrete=True`` call cannot
            # trace a model that still has unobserved continuous sites.
            samples = Predictive(
                model, posterior_samples=posterior, infer_discrete=True
            )(jax.random.PRNGKey(0))
            result.update(_discrete_marginals(samples, snapshot, discrete))
        return result


def _infer_numpyro_discrete(model, snapshot, query) -> InferenceResult:
    # ``infer_discrete=True`` makes Predictive sample the enumerated discrete
    # sites from their exact posterior (via funsor), in one vectorized pass —
    # the counterpart to a per-sample ``infer_discrete`` loop.
    samples = Predictive(model, num_samples=2000, infer_discrete=True)(
        jax.random.PRNGKey(0)
    )
    return _discrete_marginals(samples, snapshot, query)


def _discrete_marginals(samples, snapshot, names) -> dict[str, list[float]]:
    """Bincount discrete posterior samples into probability vectors."""
    result: dict[str, list[float]] = {}
    for name in names:
        node = _find_node(snapshot, name)
        card = len(node.states)
        counts = jnp.bincount(samples[name], length=card)
        probs = counts / counts.sum()
        result[name] = [float(p) for p in probs]
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
