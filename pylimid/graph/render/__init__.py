"""
Rendering backends for influence diagrams.

- `pylimid.graph.render.mermaid` emits Mermaid ``flowchart`` text.
- `pylimid.graph.render.graphviz` renders SVG through the ``dot`` binary;
  the DOT source is internal.

Both consume the backend-neutral `Scene` built here, so the two renderers
agree on what a diagram looks like and differ only in syntax.
"""
