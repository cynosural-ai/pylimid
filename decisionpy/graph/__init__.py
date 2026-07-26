"""
Graph layer: nodes and diagram representation.

Backend-agnostic. Nothing in this package imports a probabilistic
programming system; the ``dist`` callables carried by nodes return
distribution objects whose concrete type is the backend's concern.
"""

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram

__all__ = ["ChanceNode", "InfluenceDiagram"]
