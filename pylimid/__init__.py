"""
pylimid — limited-memory influence diagrams with mixed-type variables.

Build a diagram from chance, decision and utility nodes, then query it with
`infer` or find the optimal policy with `solve`::

    from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
    from pylimid import infer, solve

The graph layer (`pylimid.graph`) holds the node types and the diagram;
the inference layer (`pylimid.inference`) holds `infer`, `solve` and
their result types.
"""

from pylimid._version import __version__
from pylimid.graph import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
)
from pylimid.inference import (
    InferenceError,
    InferenceResult,
    Policy,
    Posterior,
    Solution,
    SolveMethod,
    SolverName,
    infer,
    solve,
)

__all__ = [
    "ChanceNode",
    "DecisionNode",
    "InferenceError",
    "InferenceResult",
    "InfluenceDiagram",
    "Policy",
    "Posterior",
    "Solution",
    "SolveMethod",
    "SolverName",
    "UtilityNode",
    "__version__",
    "infer",
    "solve",
]
