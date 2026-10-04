"""
Mermaid rendering for influence diagrams.

Produces a deterministic Mermaid ``flowchart`` source string from the shared
`Scene`. Mermaid is the canonical text-first rendering: no runtime
dependencies, stable output (testable and diffable), and a format that both
humans (notebooks, GitHub, chat UIs) and LLMs can read and emit.

Node shapes follow the influence-diagram convention: chance nodes are
circles, decision nodes rectangles, utility nodes diamonds. Nodes whose
consistency is not `Consistency.CONSISTENT` are tinted via ``classDef``,
and dangling parents are drawn as dashed ghost nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pylimid.graph.render._common import Scene, Shape, Style, build_scene

if TYPE_CHECKING:
    from pylimid.graph.diagram import InfluenceDiagram

__all__ = ["to_mermaid"]

_SHAPES: dict[Shape, tuple[str, str]] = {
    Shape.CIRCLE: ("((", "))"),
    Shape.RECTANGLE: ("[", "]"),
    Shape.DIAMOND: ("{", "}"),
}

_STYLES: dict[Style, str] = {
    Style.UNCONFIGURED: "fill:#e9ecef,stroke:#6c757d,color:#212529",
    Style.STALE: "fill:#fff3cd,stroke:#e0a800,color:#664d03",
    Style.DANGLING: "fill:#ffffff,stroke:#999999,stroke-dasharray: 4 4,color:#999999",
}


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
    return render(build_scene(diagram))


def render(scene: Scene) -> str:
    """Format *scene* as Mermaid ``flowchart`` source."""
    lines = ["flowchart LR"]
    for node in scene.nodes:
        open_delim, close_delim = _SHAPES[node.shape]
        cls = f":::{node.style.value}" if node.style else ""
        lines.append(
            f'    {node.id}{open_delim}"{_escape(node.name)}"{close_delim}{cls}'
        )
    for edge in scene.edges:
        arrow = "-.->" if edge.dangling else "-->"
        lines.append(f"    {edge.parent} {arrow} {edge.child}")

    used = _used_styles(scene)
    for style in used:
        lines.append(f"    classDef {style.value} {_STYLES[style]}")
    for node in scene.nodes:
        # Ghost nodes carry their class inline; only real nodes get a
        # ``class`` assignment.
        if node.style and node.style is not Style.DANGLING:
            lines.append(f"    class {node.id} {node.style.value}")

    return "\n".join(lines) + "\n"


def _used_styles(scene: Scene) -> tuple[Style, ...]:
    """The styles the scene references, in node order, without duplicates."""
    used: list[Style] = []
    for node in scene.nodes:
        if node.style and node.style not in used:
            used.append(node.style)
    return tuple(used)


def _escape(name: str) -> str:
    """The node label with embedded double quotes escaped for Mermaid."""
    return name.replace('"', "&quot;")
