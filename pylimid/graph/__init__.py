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
  *information set* (observed when deciding), ``states`` its actions.
- `UtilityNode` — a deterministic payoff ``U(parents)``; always a sink.

A Bayesian network is an `InfluenceDiagram` containing only chance nodes.
"""

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.node import Consistency, Node, NodeKind
from pylimid.graph.utility_node import UtilityNode
from pylimid.graph.validation import DiagramProblem, ProblemKind

__all__ = [
    "ChanceNode",
    "Consistency",
    "DecisionNode",
    "DiagramProblem",
    "InfluenceDiagram",
    "Node",
    "NodeKind",
    "ProblemKind",
    "UtilityNode",
]
