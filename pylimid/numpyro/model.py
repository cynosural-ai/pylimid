"""
NumPyro model factory — translate a validated snapshot into a NumPyro model.

`to_model` consumes a `DiagramSnapshot` and returns a plain NumPyro model function
the caller feeds to ``numpyro.infer``
(e.g. ``Predictive``, ``MCMC``, ``SVI``).

Unobserved discrete nodes are detected automatically via ``node.is_discrete``
and annotated with ``infer={"enumerate": "parallel"}`` so that NumPyro's NUTS
and SVI engines can marginalise them out. Observed discrete nodes skip
enumeration (the value is clamped).

Fixed decisions become degenerate observed sites; contingent decisions are
deterministic functions of their information sets. Utility nodes are skipped
— a payoff is never sampled and is never a parent.

Forward / prior-predictive sampling: pass ``observed=None`` (the default) and
feed the model to ``Predictive``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpyro
import numpyro.distributions as dist

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.utility_node import UtilityNode
from pylimid.policy import Policy, _policy_tables

__all__ = ["to_model"]


def to_model(
    snapshot: DiagramSnapshot,
    observed: dict[str, Any] | None = None,
    policy: Policy | None = None,
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

    A fixed decision becomes a degenerate observed site. A state-dependent
    decision policy is evaluated from its information-set parents as the
    model walks the diagram. A utility node is skipped entirely.

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
            absent are treated as latent. Fixed decisions are bound here.
        policy: State-dependent decision rules mapping information-set
            assignments (tuples of state indices) to action indices.

    Returns:
        A NumPyro model function taking no arguments, returning a
        ``dict[str, Any]`` of sampled (or observed) values keyed by node name.

    Raises:
        InferenceError: If a policy names a non-decision node, omits an
            information-set assignment, or contains an invalid action.
        KeyError: If a policy names a node that is not in the snapshot.
        ValueError: If a policy has a continuous information-set parent.
    """
    return _to_model(snapshot, observed or {}, _policy_tables(snapshot, policy or {}))


def _to_model(
    snapshot: DiagramSnapshot,
    observed: dict[str, Any],
    policy_tables: dict[str, jax.Array],
) -> Callable[[], dict[str, Any]]:
    """Build a model using already validated and compiled policy tables."""
    nodes = snapshot.nodes

    def model() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                if name in policy_tables:
                    parent_values = tuple(values[parent] for parent in node.parents)
                    table = policy_tables[name]
                    value = table[parent_values] if parent_values else table
                    values[name] = numpyro.deterministic(name, value)
                else:
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
