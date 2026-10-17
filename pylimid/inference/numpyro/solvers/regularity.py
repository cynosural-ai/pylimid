"""
Solvability of an influence diagram — the backward-induction gate.

A LIMID is soluble when it admits an exact solution ordering (Lauritzen
and Nilsson, 2001): an ordering of the decisions such that each decision,
with the decisions after it already resolved, has the utilities it can
influence d-separated from every other unresolved decision and its family
by its own information set. The check here is the paper's extremality
criterion read directly over the whole remaining decision set — a
decision can be placed next when its downstream utilities are d-separated
from the families of all other remaining decisions given its own family.

This deliberately diverges from pyAgrum's
``ShaferShenoyLIMIDInference.isSolvable``, which only compares decisions that
share a level of its partial order and can therefore admit diagrams whose
one-pass backward induction is not optimal. The regression tests pin the
divergence.

The resulting order is the solving order: its first decision is solved
first, and every later entry is tested against the ones already placed.
"""

from __future__ import annotations

from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.node import NodeKind

__all__ = ["is_solvable", "solvability_order"]


def is_solvable(snapshot: DiagramSnapshot) -> bool:
    """
    Check whether the graph admits an exact solution ordering.

    This is the structural check used by ``solve(method="auto")`` to choose backward
    induction. It does not check whether information sets are discrete or whether every
    assignment will appear in the samples.

    Args:
        snapshot: A validated snapshot from `InfluenceDiagram.snapshot`.

    Returns:
        ``True`` if `solvability_order` finds an ordering, including for a diagram with
        no decisions. Numerical requirements must still be satisfied to solve the
        diagram.
    """
    return solvability_order(snapshot) is not None


def solvability_order(snapshot: DiagramSnapshot) -> list[str] | None:
    """
    Find an order in which backward induction can resolve the decisions.

    Repeatedly picks a decision whose downstream utilities are d-separated from the
    families of the other unresolved decisions, given its own family. A decision's
    family consists of the decision and its parents. Previously selected decisions are
    treated as resolved.

    Args:
        snapshot: A validated snapshot from `InfluenceDiagram.snapshot`.

    Returns:
        Decision names in the order they are resolved. The first entry need not be the
        first decision in time. Returns ``None`` if no complete ordering exists, or an
        empty list if there are no decisions.
    """
    names, kinds, parents, children = _graph(snapshot)
    utilities = {name for name in names if kinds[name] is NodeKind.UTILITY}
    pending = [name for name in names if kinds[name] is NodeKind.DECISION]

    order: list[str] = []
    while pending:
        chosen = None
        for candidate in pending:
            downstream = _descendants(candidate, children) & utilities
            others: set[str] = set()
            for other in pending:
                if other != candidate:
                    others.add(other)
                    others.update(parents[other])
            info = {candidate, *parents[candidate]}
            if _d_separated(others, downstream, info, parents):
                chosen = candidate
                break
        if chosen is None:
            return None
        order.append(chosen)
        pending.remove(chosen)
    return order


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _graph(snapshot: DiagramSnapshot):
    names = [name for name, _ in snapshot.nodes]
    kinds = {name: node.kind for name, node in snapshot.nodes}
    parents = {name: tuple(node.parents) for name, node in snapshot.nodes}
    children: dict[str, list[str]] = {name: [] for name in names}
    for name in names:
        for parent in parents[name]:
            children[parent].append(name)
    return names, kinds, parents, children


def _descendants(name: str, children: dict[str, list[str]]) -> set[str]:
    """Every strict descendant of *name*."""
    found: set[str] = set()
    frontier = list(children[name])
    while frontier:
        node = frontier.pop()
        if node in found:
            continue
        found.add(node)
        frontier.extend(children[node])
    return found


# ---------------------------------------------------------------------------
# The d-separation test
# ---------------------------------------------------------------------------


def _d_separated(
    xs: set[str],
    ys: set[str],
    zs: set[str],
    parents: dict[str, tuple[str, ...]],
) -> bool:
    """
    Whether node sets *xs* and *ys* are d-separated given *zs*.

    The standard test: in the moralized ancestral graph of xs ∪ ys ∪ zs,
    xs and ys are disconnected after removing zs.
    """
    if not xs or not ys:
        return True

    keep: set[str] = set()
    frontier = list(xs | ys | zs)
    while frontier:
        node = frontier.pop()
        if node in keep:
            continue
        keep.add(node)
        frontier.extend(parents[node])

    adjacency: dict[str, set[str]] = {node: set() for node in keep}
    for node in keep:
        node_parents = [parent for parent in parents[node] if parent in keep]
        for parent in node_parents:
            adjacency[node].add(parent)
            adjacency[parent].add(node)
        for i, first in enumerate(node_parents):
            for second in node_parents[i + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    seen = set(zs)
    frontier = [node for node in xs if node not in seen]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in ys:
            return False
        frontier.extend(adjacency[node] - seen)
    return True
