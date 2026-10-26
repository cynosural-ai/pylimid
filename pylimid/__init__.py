"""
pylimid — limited-memory influence diagrams with mixed-type variables.

Build a diagram from chance, decision and utility nodes, then query it with
`infer` or fit a policy with `solve`::

    from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
    from pylimid import infer, solve

The graph layer (`pylimid.graph`) holds the node types and the diagram;
`pylimid.policy` holds policy families and decision rules. `pylimid.inference` holds
posterior inference, and `pylimid.solver` holds policy optimisation. The internal
``pylimid._numpyro`` package translates diagrams into NumPyro models and draws samples.
"""

from pylimid._version import __version__
from pylimid.errors import InferenceError, SolveError
from pylimid.graph import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
)
from pylimid.inference import InferenceResult, Posterior, infer
from pylimid.policy import (
    ActionTable,
    CategoricalPolicy,
    DecisionRule,
    LinearSoftmax,
    Policy,
    PolicyFamily,
    TabularSoftmax,
)
from pylimid.solver import Solution, solve

__all__ = [
    "ActionTable",
    "ChanceNode",
    "DecisionNode",
    "DecisionRule",
    "InferenceError",
    "InferenceResult",
    "InfluenceDiagram",
    "Policy",
    "LinearSoftmax",
    "CategoricalPolicy",
    "PolicyFamily",
    "TabularSoftmax",
    "Posterior",
    "Solution",
    "SolveError",
    "UtilityNode",
    "__version__",
    "infer",
    "solve",
]
