"""
CPT and utility-factor builders shared by the graph-native inference engines.

Both variable elimination (``decisionpy.inference.ve``) and the
influence-diagram solvers (``decisionpy.inference.id``) consume a validated
:class:`~decisionpy.graph.diagram.Snapshot` and need to turn a node's
configurable callable into a :class:`~decisionpy.inference._factor.Factor` over
its variables:

- a chance node's ``dist`` becomes the CPT factor ``P(node | parents)``;
- a utility node's ``values`` becomes the utility factor ``U(parents)``.

The two builders share the same enumeration pattern (walk every assignment of
the parent variables, invoke the callable with keyword arguments keyed by
parent *name*), so they live together here. ``cardinalities`` supplies the
per-variable domain sizes both builders need.
"""

from __future__ import annotations

import math
from itertools import product

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference._factor import Factor

__all__ = ["cardinalities", "cpt", "utility_factor"]


def cardinalities(snapshot: Snapshot) -> dict[str, int]:
    """
    Map each discrete variable in *snapshot* to its number of states.

    Chance and decision nodes contribute their ``states`` length; utility
    nodes are skipped (a payoff is a deterministic scalar, never sampled).

    Raises:
        TypeError: If any chance or decision node has no declared ``states``
            (the diagram is not all-categorical).
    """
    card: dict[str, int] = {}
    for name, node in snapshot.nodes:
        if isinstance(node, UtilityNode):
            continue
        states = getattr(node, "states", None)
        if states is None:
            raise TypeError(
                f"Node {name!r} has no declared states; a discrete diagram "
                f"is required."
            )
        card[name] = len(states)
    return card


def cpt(node: ChanceNode, card: dict[str, int]) -> Factor:
    """
    Build the CPT factor ``P(node | parents(node))`` from ``node.dist``.

    Args:
        node: A configured discrete chance node.
        card: Variable-name to domain-size map (from :func:`cardinalities`).

    Returns:
        A factor over ``node.parents + [node.name]`` whose entry for each
        assignment is the corresponding row of the node's distribution.

    Raises:
        RuntimeError: If ``node.dist`` yields no probability vector (neither a
            ``probs`` attribute nor a working ``log_prob``).
    """
    parent_vars = list(node.parents)
    parent_cards = [card[p] for p in parent_vars]
    node_card = card[node.name]

    values: list[float] = []
    for parent_vals in product(*[range(c) for c in parent_cards]):
        kwargs = dict(zip(node.parents, parent_vals, strict=True))
        prob_vec = _extract_probs(node, kwargs, node_card)
        if prob_vec is None:
            raise RuntimeError(
                f"Could not extract probability vector from node "
                f"{node.name!r}. Make sure its ``dist`` returns a "
                f"Categorical or a distribution with a ``probs`` attribute."
            )
        values.extend(prob_vec)

    return Factor(
        variables=parent_vars + [node.name],
        card=tuple(parent_cards + [node_card]),
        values=values,
    )


def utility_factor(node: UtilityNode, card: dict[str, int]) -> Factor:
    """
    Build the utility factor ``U(parents(node))`` from ``node.values``.

    Args:
        node: A configured utility node (``values`` set and consistent).
        card: Variable-name to domain-size map (from :func:`cardinalities`).

    Returns:
        A factor over ``node.parents`` whose entry for each assignment is the
        payoff ``node.values(**parents)``.
    """
    parent_vars = list(node.parents)
    parent_cards = [card[p] for p in parent_vars]
    assert node.values is not None

    values: list[float] = []
    for parent_vals in product(*[range(c) for c in parent_cards]):
        kwargs = dict(zip(parent_vars, parent_vals, strict=True))
        values.append(float(node.values(**kwargs)))

    return Factor(
        variables=parent_vars,
        card=tuple(parent_cards),
        values=values,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_probs(
    node: ChanceNode,
    parent_kwargs: dict[str, object],
    card: int,
) -> list[float] | None:
    """Call ``node.dist(**parent_kwargs)`` and extract a probability vector."""
    assert node.dist is not None
    dist = node.dist(**parent_kwargs)
    probs = getattr(dist, "probs", None)
    if probs is not None:
        return [float(p) for p in probs]
    # Fallback: materialise via log_prob for each outcome.
    try:
        log_probs = [dist.log_prob(i) for i in range(card)]
        return [math.exp(float(lp)) for lp in log_probs]
    except Exception:
        return None
