"""
Unified inference entry-point.

::

    from decisionpy.inference import infer

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": [0.456, 0.544]}  (exact posteriors for discrete BNs)
"""

from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp

from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import NodeKind
from decisionpy.inference.numpyro import samples as numpyro_samples
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
    # ``samples()`` returns raw draws; ``infer()`` normalizes them to the
    # per-type contract: probability vectors for discrete variables, plain
    # lists of draws for continuous ones.
    draws = numpyro_samples(snapshot, observed=observed, query=query)
    result: InferenceResult = {}
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is NodeKind.CHANCE and node.is_discrete:
            counts = jnp.bincount(draws[name], length=len(node.states))
            probs = counts / counts.sum()
            result[name] = [float(p) for p in probs]
        else:
            result[name] = [float(x) for x in draws[name]]
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
