"""
Mermaid rendering for influence diagrams.

Produces a deterministic Mermaid ``flowchart`` source string for an
:class:`~decisionpy.graph.diagram.InfluenceDiagram`. Mermaid is the canonical
text-first rendering: no runtime dependencies, stable output (testable and
diffable), and a format that both humans (notebooks, GitHub, chat UIs) and
LLMs can read and emit.

Node shapes follow the influence-diagram convention: chance nodes are
circles, decision nodes rectangles, utility nodes diamonds.

The renderer works on the live, possibly incomplete workspace: dangling
parents are drawn as dashed ghost nodes with dashed edges, and nodes whose
consistency is not :attr:`~decisionpy.graph.node.Consistency.CONSISTENT` are
tinted via ``classDef`` so a reader sees at a glance where the diagram is
unfinished.
"""

import re

from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import Consistency, NodeKind

_SHAPES: dict[NodeKind, tuple[str, str]] = {
    NodeKind.CHANCE: ("((", "))"),
    NodeKind.DECISION: ("[", "]"),
    NodeKind.UTILITY: ("{", "}"),
}

_CLASS_STYLES: dict[str, str] = {
    "unconfigured": "fill:#e9ecef,stroke:#6c757d,color:#212529",
    "stale": "fill:#fff3cd,stroke:#e0a800,color:#664d03",
    "dangling": "fill:#ffffff,stroke:#999999,stroke-dasharray: 4 4,color:#999999",
}

_CONSISTENCY_CLASS = {
    Consistency.UNCONFIGURED: "unconfigured",
    Consistency.STALE: "stale",
}

_ID_STRIP = re.compile(r"[^A-Za-z0-9_]")


def to_mermaid(diagram: InfluenceDiagram) -> str:
    """
    Render ``diagram`` as a Mermaid flowchart source string.

    Output is deterministic: nodes are emitted in topological order (parents
    before children) and edges follow each child's declared parent order, so
    the same diagram always renders the same source. Rendering does not
    require a valid diagram — dangling parents and non-consistent nodes are
    drawn, not rejected.

    Args:
        diagram: The diagram to render. May be incomplete.

    Returns:
        Mermaid ``flowchart`` source. Paste into a Mermaid renderer (GitHub,
        Jupyter, mermaid.live) to view it.
    """
    lines = ["flowchart LR"]
    ids: dict[str, str] = {}
    dangling = _dangling_parents(diagram)
    try:
        order = diagram.topological_sort()
    except ValueError:
        order = diagram.names

    # Node declarations: real nodes in topological order, then ghost nodes
    # for every dangling parent reference.
    for name in order:
        node = diagram[name]
        ids[name] = _unique_id(name, ids)
        open_delim, close_delim = _SHAPES[node.kind]
        cls = _class_attr(node.consistency)
        lines.append(f'    {ids[name]}{open_delim}"{_label(name)}"{close_delim}{cls}')
    for name in dangling:
        ids[name] = _unique_id(name, ids)
        lines.append(f'    {ids[name]}["{_label(name)}"]:::dangling')

    # Edge declarations. Edges to dangling parents are dashed and skip the
    # diagram's cycle-free wiring (a dangling parent is not a real node yet).
    for name in order:
        for parent in diagram[name].parents:
            arrow = "-.->" if parent in dangling else "-->"
            lines.append(f"    {ids[parent]} {arrow} {ids[name]}")

    # classDef styles, only for classes actually used.
    for cls in _used_classes(diagram, dangling):
        lines.append(f"    classDef {cls} {_CLASS_STYLES[cls]}")
    for name in order:
        cls = _consistency_class(diagram[name].consistency)
        if cls:
            lines.append(f"    class {ids[name]} {cls}")

    return "\n".join(lines) + "\n"


def _dangling_parents(diagram: InfluenceDiagram) -> tuple[str, ...]:
    """Names listed as parents but not registered in the diagram, deduplicated."""
    dangling: set[str] = set()
    for name in diagram.names:
        for parent in diagram[name].parents:
            if parent not in diagram:
                dangling.add(parent)
    return tuple(sorted(dangling))


def _unique_id(name: str, ids: dict[str, str]) -> str:
    """A Mermaid-safe node id unique among the ids already allocated."""
    base = _ID_STRIP.sub("_", name) or "n"
    candidate = base
    taken = set(ids.values())
    counter = 1
    while candidate in taken:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


def _label(name: str) -> str:
    """The node label with embedded double quotes escaped for Mermaid."""
    return name.replace('"', "&quot;")


def _consistency_class(consistency: Consistency) -> str:
    """The Mermaid class name for a non-consistent node, else empty string."""
    return _CONSISTENCY_CLASS.get(consistency, "")


def _class_attr(consistency: Consistency) -> str:
    """The Mermaid ``:::class`` suffix for a non-consistent node, else empty."""
    cls = _consistency_class(consistency)
    return f":::{cls}" if cls else ""


def _used_classes(
    diagram: InfluenceDiagram, dangling: tuple[str, ...]
) -> tuple[str, ...]:
    """The classDef names the rendered output actually references, in order."""
    used: list[str] = []
    for name in diagram.names:
        cls = _consistency_class(diagram[name].consistency)
        if cls and cls not in used:
            used.append(cls)
    if dangling and "dangling" not in used:
        used.append("dangling")
    return tuple(used)
