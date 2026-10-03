"""
NumPyro model factory — translate a validated snapshot into a NumPyro model.

Internal module. Public convenience functions that wrap this are in
`pylimid.inference.numpyro`.

This is the bridge between the backend-agnostic graph layer and a concrete
probabilistic programming system. It consumes a `Snapshot` and returns a plain
NumPyro model function the caller feeds to ``numpyro.infer``
(e.g. ``Predictive``, ``MCMC``, ``SVI``).

NumPyro is a declared runtime dependency of pylimid (nothing in
`pylimid.graph` imports it — the graph layer works on plain callables,
but the translator itself needs it).

Unobserved discrete nodes are detected automatically via ``node.is_discrete``
and annotated with ``infer={"enumerate": "parallel"}`` so that NumPyro's NUTS
and SVI engines can marginalise them out. Observed discrete nodes skip
enumeration (the value is clamped).

Bound decisions (clamped by a policy) become degenerate observed sites; utility
nodes are skipped — a payoff is never sampled and is never a parent.

Forward / prior-predictive sampling: pass ``observed=None`` (the default) and
feed the model to ``Predictive``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpyro
import numpyro.distributions as dist

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import Snapshot
from pylimid.graph.utility_node import UtilityNode

__all__ = ["to_model"]


def to_model(
    snapshot: Snapshot,
    observed: dict[str, Any] | None = None,
) -> Callable[[], dict[str, Any]]:
    """
    Build a NumPyro model function from a validated snapshot.

    The returned ``model()`` walks the snapshot's nodes in topological order
    (parents before children — guaranteed by the snapshot), resolves each node's
    parents by name against the upstream sampled values, and emits one
    ``numpyro.sample`` site per chance node.

    Unobserved discrete chance nodes are auto-detected and annotated with
    ``infer={"enumerate": "parallel"}`` so that NUTS and SVI can marginalise
    them out. Observed nodes skip enumeration and are clamped via ``obs=``.
    Continuous nodes are left for the inference engine.

    A decision node becomes a degenerate observed site at its bound value —
    every decision must appear in *observed* (``infer(policy=...)`` merges the
    policy there). A utility node is skipped entirely.

    Note:
        Under NumPyro's parallel enumeration, the parent values passed to a
        ``dist`` callable may carry an enumeration dimension. Index
        multi-parent probability tables in a single indexing operation
        (``T[a, b]``) rather than chained (``T[a][b]``) — NumPyro computes
        wrong posterior weights for chained indexing.

    Args:
        snapshot: A validated, topologically-ordered view of an influence
            diagram.
        observed: Mapping of node names to observed values. Chance nodes
            present here are conditioned on their given value; chance nodes
            absent are treated as latent; decisions must always be present.
            Pass ``None`` or ``{}`` for forward sampling of a
            chance-node-only diagram.

    Returns:
        A NumPyro model function taking no arguments, returning a
        ``dict[str, Any]`` of sampled (or observed) values keyed by node name.
    """
    nodes = snapshot.nodes
    observed = observed or {}

    def model() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                value = observed[name]
                values[name] = numpyro.sample(name, dist.Delta(value), obs=value)
                continue
            assert isinstance(node, ChanceNode)
            parent_values = {parent: values[parent] for parent in node.parents}
            assert node.dist is not None
            fn = node.dist(**parent_values)
            if name in observed:
                values[name] = numpyro.sample(name, fn, obs=observed[name])
            elif node.is_discrete:
                values[name] = numpyro.sample(name, fn, infer={"enumerate": "parallel"})
            else:
                values[name] = numpyro.sample(name, fn)
        return values

    return model
