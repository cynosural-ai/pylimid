"""
Solvability of an influence diagram — the backward-induction gate.

A LIMID is solvable when its decisions can be ordered so that backward
induction is valid: every decision that could influence another's subproblem
is either observed or d-separated by the information set. The check mirrors
the criterion aGrUM's Shafer-Shenoy LIMID inference implements (pyAgrum's
``ShaferShenoyLIMIDInference.isSolvable()``), in two stages:

1. Level the nodes by distance from the utilities: ``level(utility) = 0``
   and ``level(x) = max over children c of level(c) + (1 if c is a
   decision)``, leaves included. Decisions land in levels, with the
   decisions closest to the utilities at level 0.
2. Process the levels from 0 upward. Within a level, repeatedly pick a
   decision whose downstream utilities are d-separated from the remaining
   same-level decisions (and their information sets) given its own
   information set. A level that cannot be emptied means the diagram is not
   solvable.

The resulting order is the solving order: its first decision is the one
solved first (the one closest to the utilities).
"""

from __future__ import annotations

from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.node import NodeKind

__all__ = ["is_solvable", "solvability_order"]


def is_solvable(snapshot: Snapshot) -> bool:
    """Whether *snapshot* admits a valid backward-induction order."""
    return solvability_order(snapshot) is not None


def solvability_order(snapshot: Snapshot) -> list[str] | None:
    """
    The decisions in solving order, or None if the diagram is not solvable.

    The first decision in the list is solved first — it is the decision
    closest to the utilities. Only the boolean matters for the gate;
    the order is what a backward-induction solver processes.
    """
    names, kinds, parents, children = _graph(snapshot)
    utilities = {name for name in names if kinds[name] is NodeKind.UTILITY}
    levels = _decision_levels(names, kinds, utilities, children)
    return _order_levels(levels, utilities, parents, children)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _graph(snapshot: Snapshot):
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
# Stage 1 — level the nodes by distance from the utilities
# ---------------------------------------------------------------------------


def _decision_levels(
    names: list[str],
    kinds: dict[str, NodeKind],
    utilities: set[str],
    children: dict[str, list[str]],
) -> dict[int, list[str]]:
    """
    Decisions grouped by level; level 0 is closest to the utilities.

    Nodes are visited in reverse topological order (children before
    parents — the snapshot is topologically ordered), so every node's
    children carry a level by the time it is visited.
    """
    level = dict.fromkeys(utilities, 0)
    levels: dict[int, list[str]] = {}
    for name in reversed(names):
        if name in utilities:
            continue
        level[name] = max(
            (
                level[child] + (1 if kinds[child] is NodeKind.DECISION else 0)
                for child in children[name]
            ),
            default=0,
        )
        if kinds[name] is NodeKind.DECISION:
            levels.setdefault(level[name], []).append(name)
    return levels


# ---------------------------------------------------------------------------
# Stage 2 — order each level with the d-separation test
# ---------------------------------------------------------------------------


def _order_levels(
    levels: dict[int, list[str]],
    utilities: set[str],
    parents: dict[str, tuple[str, ...]],
    children: dict[str, list[str]],
) -> list[str] | None:
    order: list[str] = []
    for level in sorted(levels):
        remaining = list(levels[level])
        while remaining:
            chosen = None
            for candidate in remaining:
                downstream = _descendants(candidate, children) & utilities
                others: set[str] = set()
                for other in remaining:
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
            remaining.remove(chosen)
    return order


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
