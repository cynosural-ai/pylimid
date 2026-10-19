"""
Backend-neutral scene shared by the influence-diagram renderers.

A `Scene` is an influence diagram flattened for rendering: the real nodes in
topological order (parents before children), a dashed ghost node for every
dangling parent reference, the backend-neutral `Shape` and `Style` of each
node, and the edges. The renderers map `Shape` and `Style` to their own
syntax; keeping the semantics here means Mermaid and Graphviz cannot diverge
on what a diagram looks like.

Rendering works on the live, possibly incomplete workspace — it never
validates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pylimid.graph.node import Consistency, NodeKind

if TYPE_CHECKING:
    from pylimid.graph.diagram import InfluenceDiagram

__all__ = ["Edge", "Node", "Scene", "Shape", "Style", "build_scene"]


class Shape(Enum):
    """The backend-neutral shape of a node."""

    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    DIAMOND = "diamond"


class Style(Enum):
    """The backend-neutral emphasis of a node."""

    UNCONFIGURED = "unconfigured"
    STALE = "stale"
    DANGLING = "dangling"


_SHAPES = {
    NodeKind.CHANCE: Shape.CIRCLE,
    NodeKind.DECISION: Shape.RECTANGLE,
    NodeKind.UTILITY: Shape.DIAMOND,
}

_STYLES = {
    Consistency.UNCONFIGURED: Style.UNCONFIGURED,
    Consistency.STALE: Style.STALE,
}

_ID_STRIP = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class Node:
    """
    One renderable node.

    Attributes:
        name: The node's original name, for display.
        id: A sanitized identifier unique within the scene.
        shape: The backend-neutral shape.
        style: The emphasis, or ``None`` for a consistent node.
    """

    name: str
    id: str
    shape: Shape
    style: Style | None


@dataclass(frozen=True)
class Edge:
    """
    One renderable edge, by sanitized node id.

    Attributes:
        parent: The parent node's id.
        child: The child node's id.
        dangling: Whether the parent is a ghost node.
    """

    parent: str
    child: str
    dangling: bool


@dataclass(frozen=True)
class Scene:
    """
    A diagram flattened into renderable nodes and edges.

    Attributes:
        nodes: Real nodes in topological order, then ghost nodes for
            dangling parents.
        edges: One entry per declared parent, in topological order.
    """

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def build_scene(diagram: InfluenceDiagram) -> Scene:
    """
    Flatten *diagram* into the shared render scene.

    Args:
        diagram: The diagram to render. May be incomplete: dangling parents
            become ghost nodes and a cyclic workspace falls back to
            insertion order.

    Returns:
        The backend-neutral `Scene`.
    """
    dangling = _dangling_parents(diagram)
    try:
        order = diagram.topological_sort()
    except ValueError:
        order = diagram.names

    ids: dict[str, str] = {}
    nodes: list[Node] = []
    for name in order:
        node = diagram[name]
        ids[name] = _unique_id(name, ids)
        nodes.append(
            Node(name, ids[name], _shape_of(node.kind), _style_of(node.consistency))
        )
    for name in dangling:
        ids[name] = _unique_id(name, ids)
        nodes.append(Node(name, ids[name], Shape.RECTANGLE, Style.DANGLING))

    edges = [
        Edge(ids[parent], ids[name], parent in dangling)
        for name in order
        for parent in diagram[name].parents
    ]
    return Scene(tuple(nodes), tuple(edges))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_of(kind: NodeKind) -> Shape:
    """The shape for *kind*; raises on a kind without a shape."""
    return _SHAPES[kind]


def _style_of(consistency: Consistency) -> Style | None:
    """The emphasis for *consistency*, or ``None`` for a consistent node."""
    if consistency is Consistency.CONSISTENT:
        return None
    return _STYLES[consistency]


def _dangling_parents(diagram: InfluenceDiagram) -> tuple[str, ...]:
    """Names listed as parents but not registered in the diagram, deduplicated."""
    dangling: set[str] = set()
    for name in diagram.names:
        for parent in diagram[name].parents:
            if parent not in diagram:
                dangling.add(parent)
    return tuple(sorted(dangling))


def _unique_id(name: str, ids: dict[str, str]) -> str:
    """A renderer-safe node id unique among the ids already allocated."""
    base = _ID_STRIP.sub("_", name) or "n"
    candidate = base
    taken = set(ids.values())
    counter = 1
    while candidate in taken:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate
