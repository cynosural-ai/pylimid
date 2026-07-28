"""
Graph layer: nodes and diagram representation.

Backend-agnostic. Nothing in this package imports a probabilistic
programming system; the ``dist`` callables carried by chance nodes return
distribution objects whose concrete type is the backend's concern.

Node types
----------
Three node kinds share a common base (:class:`Node`):

- :class:`ChanceNode` — a random variable, ``P(name | parents)``.
- :class:`DecisionNode` — a variable the agent controls; ``parents`` are its
  *information set* (observed when deciding), ``states`` its actions.
- :class:`UtilityNode` — a deterministic payoff ``U(parents)``; always a sink.

A Bayesian network is a diagram containing only chance nodes.
"""

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import Consistency, Node, NodeKind
from decisionpy.graph.utility_node import UtilityNode

__all__ = [
    "ChanceNode",
    "Consistency",
    "DecisionNode",
    "InfluenceDiagram",
    "Node",
    "NodeKind",
    "UtilityNode",
]
