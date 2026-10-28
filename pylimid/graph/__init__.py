"""
Graph layer: nodes and diagram representation.

Backend-agnostic. Nothing in this package imports a probabilistic
programming system; the ``dist`` callables carried by chance nodes return
distribution objects whose concrete type is the backend's concern.

Node types
----------
Three node kinds share a common base (`Node`):

- `ChanceNode` — a random variable, ``P(name | parents)``.
- `DecisionNode` — a variable the agent controls; ``parents`` are its
  *information set* (observed when deciding), and ``states`` or ``action_space``
  declares its actions.
- `UtilityNode` — a deterministic payoff ``U(parents)``; always a sink.

A Bayesian network is an `InfluenceDiagram` containing only chance nodes.
"""

from pylimid.graph.action_space import Interval, Real
from pylimid.graph.diagram import (
    DiagramProblem,
    DiagramProblemKind,
    DiagramSnapshot,
    InfluenceDiagram,
)
from pylimid.graph.nodes import (
    ChanceNode,
    Consistency,
    DecisionNode,
    Node,
    NodeKind,
    UtilityNode,
)

__all__ = [
    "ChanceNode",
    "Consistency",
    "DecisionNode",
    "DiagramProblem",
    "DiagramProblemKind",
    "DiagramSnapshot",
    "InfluenceDiagram",
    "Interval",
    "Node",
    "NodeKind",
    "Real",
    "UtilityNode",
]
