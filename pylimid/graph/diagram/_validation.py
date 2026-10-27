"""Structural validation and node configuration checks for influence diagrams."""

from collections.abc import Mapping

from pylimid.graph.diagram._topology import children_of, find_cycle, kahn_order
from pylimid.graph.diagram.problems import DiagramProblem, DiagramProblemKind
from pylimid.graph.nodes import Consistency, Node


def validate_nodes(nodes: Mapping[str, Node]) -> list[DiagramProblem]:
    """
    Check graph structure and aggregate node configuration problems.

    Inspect missing parents, callable signatures, utility sinks, and cycles without
    executing distribution or utility functions.

    Args:
        nodes: Registered nodes keyed by name. May be incomplete or inconsistent.

    Returns:
        The detected problems in check order, or an empty list if the checks pass.
    """
    problems: list[DiagramProblem] = []

    # Dangling parents: a declared parent not present in the diagram.
    for name, node in nodes.items():
        for parent in node.parents:
            if parent not in nodes:
                problems.append(
                    DiagramProblem(
                        kind=DiagramProblemKind.DANGLING_PARENT,
                        node=name,
                        message=(
                            f"Parent {parent!r} of {name!r} is not in the diagram."
                        ),
                    )
                )

    # Per-node consistency: inference needs every node CONSISTENT. The
    # wording is owned by the node (its configurable field differs by kind
    # — `dist`, `values`, action `states`), so the message comes from the
    # node's own `consistency_message`.
    for name, node in nodes.items():
        state = node.consistency
        if state is Consistency.UNCONFIGURED:
            problems.append(
                DiagramProblem(
                    kind=DiagramProblemKind.UNCONFIGURED,
                    node=name,
                    message=node.consistency_message(state),
                )
            )
        elif state is Consistency.STALE:
            problems.append(
                DiagramProblem(
                    kind=DiagramProblemKind.STALE,
                    node=name,
                    message=node.consistency_message(state),
                )
            )

    # A utility node is a payoff, so it must be terminal.
    for name, node in nodes.items():
        if node.is_sink and children_of(nodes, name):
            problems.append(
                DiagramProblem(
                    kind=DiagramProblemKind.UTILITY_NOT_SINK,
                    node=name,
                    message=(
                        f"Utility node {name!r} must be terminal but has "
                        f"children {children_of(nodes, name)!r}."
                    ),
                )
            )

    blocked = set(nodes) - set(kahn_order(nodes))
    if blocked:
        cycle = find_cycle(nodes, blocked)
        path = " -> ".join(repr(name) for name in (*cycle, cycle[0]))
        problems.append(
            DiagramProblem(
                kind=DiagramProblemKind.CYCLE,
                node=cycle[0],
                message=f"Diagram contains a cycle: {path}.",
            )
        )

    return problems
