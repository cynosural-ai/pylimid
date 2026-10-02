"""
decisionpy — limited-memory influence diagrams with mixed-type variables.

Build a diagram from chance, decision and utility nodes, then query it with
`infer` or find the optimal policy with `solve`::

    from decisionpy import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
    from decisionpy import infer, solve

The graph layer (`decisionpy.graph`) holds the node types and the diagram;
the inference layer (`decisionpy.inference`) holds `infer`, `solve` and
their result types.
"""

from decisionpy.graph import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
)
from decisionpy.inference import (
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

__version__ = "0.1.0"

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
