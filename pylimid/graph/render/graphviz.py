"""
Graphviz rendering for influence diagrams.

`to_svg` renders a `Scene` to SVG by piping the DOT source through the
Graphviz ``dot`` binary. The DOT source (`_to_dot`) is an internal detail:
deterministic and dependency-free, but not part of the public surface —
Mermaid is the text rendering.

The only requirement is the ``dot`` binary on ``PATH`` (a system install:
``brew install graphviz``, ``apt install graphviz``). `available` reports
whether it is there, so `InfluenceDiagram._repr_svg_` can fall back to the
plain repr instead of failing the display; an explicit `to_svg` call fails
naturally when it is missing.

Node shapes follow the influence-diagram convention: chance nodes are
circles, decision nodes boxes, utility nodes diamonds; dangling parents are
dashed ghost nodes and non-consistent nodes are tinted, mirroring the
Mermaid renderer.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING, NamedTuple

from pylimid.graph.render._common import Scene, Shape, Style, build_scene

if TYPE_CHECKING:
    from pylimid.graph.diagram import InfluenceDiagram

__all__ = ["available", "to_svg"]

_SHAPES: dict[Shape, str] = {
    Shape.CIRCLE: "circle",
    Shape.RECTANGLE: "box",
    Shape.DIAMOND: "diamond",
}


class _NodeStyle(NamedTuple):
    """The DOT attributes emphasizing a non-consistent node."""

    fill: str
    stroke: str
    text: str
    dashed: bool = False


_STYLES: dict[Style, _NodeStyle] = {
    Style.UNCONFIGURED: _NodeStyle("#e9ecef", "#6c757d", "#212529"),
    Style.STALE: _NodeStyle("#fff3cd", "#e0a800", "#664d03"),
    Style.DANGLING: _NodeStyle("#ffffff", "#999999", "#999999", dashed=True),
}


def available() -> bool:
    """Whether the Graphviz ``dot`` binary is on ``PATH``."""
    return shutil.which("dot") is not None


def to_svg(diagram: InfluenceDiagram) -> str:
    """
    Render ``diagram`` as an SVG string with the Graphviz ``dot`` binary.

    Args:
        diagram: The diagram to render. May be incomplete.

    Returns:
        SVG source.

    Raises:
        FileNotFoundError: If the ``dot`` binary is not on ``PATH``.
        subprocess.CalledProcessError: If ``dot`` rejects the graph.
    """
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=_to_dot(diagram),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _to_dot(diagram: InfluenceDiagram) -> str:
    """
    The internal DOT source for *diagram*, consumed by `to_svg`.

    Deterministic, mirroring `to_mermaid`: nodes in topological order,
    dangling parents as dashed ghost nodes, non-consistent nodes tinted.
    Rendering does not require a valid diagram.
    """
    return render(build_scene(diagram))


def render(scene: Scene) -> str:
    """Format *scene* as Graphviz DOT source."""
    lines = ["digraph {", "    rankdir=LR;"]
    for node in scene.nodes:
        attrs = [f'label="{_escape(node.name)}"', f"shape={_SHAPES[node.shape]}"]
        if node.style:
            style = _STYLES[node.style]
            attrs.append('style="filled,dashed"' if style.dashed else "style=filled")
            attrs.append(f'fillcolor="{style.fill}"')
            attrs.append(f'color="{style.stroke}"')
            attrs.append(f'fontcolor="{style.text}"')
        lines.append(f"    {node.id} [{', '.join(attrs)}];")
    for edge in scene.edges:
        suffix = " [style=dashed]" if edge.dangling else ""
        lines.append(f"    {edge.parent} -> {edge.child}{suffix};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _escape(name: str) -> str:
    """The node label escaped inside a DOT double-quoted string."""
    return name.replace("\\", "\\\\").replace('"', '\\"')
