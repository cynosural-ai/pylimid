"""
pylimid — limited-memory influence diagrams with mixed-type variables.

Build a diagram from chance, decision and utility nodes, then query it with
`infer` or find the optimal policy with `solve`::

    from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
    from pylimid import infer, solve

The graph layer (`pylimid.graph`) holds the node types and the diagram;
`pylimid.inference` holds posterior inference, and `pylimid.solver` holds
policy optimisation. `pylimid.numpyro` translates diagrams into NumPyro models
and draws samples.
"""

from pylimid._version import __version__
from pylimid.errors import InferenceError
from pylimid.graph import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
)
from pylimid.inference import InferenceResult, Posterior, infer
from pylimid.policy import Policy
from pylimid.policy_family import (
    LinearSoftmax,
    SoftmaxPolicy,
    TabularPolicy,
    TabularSoftmax,
)
from pylimid.solver import Solution, SolveMethod, SolverName, solve

__all__ = [
    "ChanceNode",
    "DecisionNode",
    "InferenceError",
    "InferenceResult",
    "InfluenceDiagram",
    "Policy",
    "LinearSoftmax",
    "SoftmaxPolicy",
    "TabularSoftmax",
    "TabularPolicy",
    "Posterior",
    "Solution",
    "SolveMethod",
    "SolverName",
    "UtilityNode",
    "__version__",
    "infer",
    "solve",
]
