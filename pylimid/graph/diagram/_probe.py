"""Runtime checks for categorical parents whose effects may be missing."""

from collections.abc import Mapping

from pylimid.graph.diagram.problems import DiagramProblem, DiagramProblemKind
from pylimid.graph.nodes.chance_node import ChanceNode
from pylimid.graph.nodes.decision_node import DecisionNode
from pylimid.graph.nodes.node import Node
from pylimid.graph.nodes.utility_node import UtilityNode


def probe_discrete_parents(nodes: Mapping[str, Node]) -> list[DiagramProblem]:
    """
    Find categorical parents that have no effect at the tested inputs.

    Execute configured chance and utility functions with each categorical parent
    varied across its state indices and all other parents fixed at zero. Errors from
    the functions propagate. Numerical parents are not varied.

    Args:
        nodes: Registered nodes keyed by name. May include incomplete nodes and
            dangling parent references.

    Returns:
        One warning per node and categorical parent whose output is unchanged at the
        tested inputs. Unconfigured functions are skipped. A finding does not
        establish that the parent has no effect at other inputs.
    """
    problems: list[DiagramProblem] = []
    for name, node in nodes.items():
        if isinstance(node, ChanceNode):
            callable_ = node.dist
        elif isinstance(node, UtilityNode):
            callable_ = node.values
        else:
            continue
        if callable_ is None:
            continue  # UNCONFIGURED — validate() reports it
        for parent in node.parents:
            parent_node = nodes.get(parent)
            if not isinstance(parent_node, (ChanceNode, DecisionNode)):
                continue  # dangling or not a chance or decision node
            states = parent_node.states
            if states is None:
                continue
            fixed = {p: 0 for p in node.parents if p != parent}
            fingerprints = {
                _distribution_fingerprint(callable_(**{**fixed, parent: value}))
                for value in range(len(states))
            }
            if len(fingerprints) == 1:
                problems.append(
                    DiagramProblem(
                        kind=DiagramProblemKind.DIST_IGNORES_PARENT,
                        node=name,
                        message=(
                            f"The callable for node {name!r} returns the same "
                            f"distribution for every value of parent "
                            f"{parent!r}; {parent} appears to have no "
                            f"effect. Warning: a deliberately independent "
                            f"node looks identical."
                        ),
                    )
                )
    return problems


def _distribution_fingerprint(output) -> tuple[object, ...]:
    """
    A comparable snapshot of a callable's output.

    NumPyro distributions expose their defining parameters uniformly via
    ``get_args()`` (``{'loc': ..., 'scale': ...}``), so two distributions
    are compared exactly: equal parameters mean equal distributions. A
    plain scalar (a utility ``values`` payoff) is its own fingerprint.
    Anything else fails loudly — the library's contract is that ``dist``
    returns a distribution object and ``values`` returns a number. Equal
    fingerprints are treated as "the same output" by
    `InfluenceDiagram.probe_discrete_parents`.
    """
    get_args = getattr(output, "get_args", None)
    if get_args is not None:
        args = get_args()
        if args:
            return tuple(
                (name, _comparable(value)) for name, value in sorted(args.items())
            )
    return (float(output),)


def _comparable(value) -> object:
    """Convert an array-like parameter to a hashable, comparable form."""
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        return _freeze(tolist())
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _freeze(value):
    """Convert nested lists to nested tuples so they can be hashed."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
