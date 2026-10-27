"""Derived adjacency, topological ordering, and cycle discovery."""

from collections.abc import Mapping

from pylimid.graph.nodes import Node


def topological_sort(nodes: Mapping[str, Node]) -> tuple[str, ...]:
    """
    Order registered nodes with parents before children.

    Args:
        nodes: Registered nodes keyed by name. Missing parents are ignored.

    Returns:
        Node names in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    order = kahn_order(nodes)
    if len(order) != len(nodes):
        raise ValueError(
            "Diagram has a cycle; cannot topologically sort. "
            "Run validate() to locate it."
        )
    return tuple(order)


def children_of(nodes: Mapping[str, Node], name: str) -> tuple[str, ...]:
    """
    Find registered nodes that declare a name as a parent.

    Args:
        nodes: Registered nodes keyed by name.
        name: Parent name to look up. It need not be registered.

    Returns:
        Child names in mapping order, or an empty tuple if none refer to the name.
    """
    return tuple(
        other_name for other_name, other in nodes.items() if name in other.parents
    )


def kahn_order(nodes: Mapping[str, Node]) -> list[str]:
    """
    Apply Kahn's algorithm, stopping where a cycle blocks it.

    Args:
        nodes: Registered nodes keyed by name. Missing parents are ignored.

    Returns:
        Every node in topological order when acyclic. Otherwise, nodes on cycles or
        downstream of them are omitted.
    """
    indegree: dict[str, int] = dict.fromkeys(nodes, 0)
    children = {name: [] for name in nodes}
    for name, node in nodes.items():
        for parent in node.parents:
            if parent in nodes:
                children[parent].append(name)
                indegree[name] += 1
    queue = [name for name, degree in indegree.items() if degree == 0]
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order


def find_cycle(nodes: Mapping[str, Node], blocked: set[str]) -> tuple[str, ...]:
    """
    Find one cycle among nodes Kahn's algorithm could not order.

    Every blocked node has a blocked parent, so walking up parents must eventually
    repeat a node.

    Args:
        nodes: Registered nodes keyed by name.
        blocked: Non-empty set of nodes omitted by `kahn_order`.

    Returns:
        Cycle node names in arc order, starting from the repeated node.
    """
    current = min(blocked)
    path: list[str] = []
    position: dict[str, int] = {}
    while current not in position:
        position[current] = len(path)
        path.append(current)
        current = next(p for p in nodes[current].parents if p in blocked)
    return tuple(reversed(path[position[current] :]))
