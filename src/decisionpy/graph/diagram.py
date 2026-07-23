"""
Influence diagram — the graph container.

This is the mutable workspace design settled in
``docs/diagram_mutable_design.md`` and adopted as the library's direction in
``docs/17_07_2026_mutability_design_decision.md``.

Mutable workspace, validated gate
---------------------------------
The diagram is a workspace that an external author — a script, a UI, or an LLM
driving the model over a tool interface — edits incrementally: drop a node
here, wire an edge there, fill in a distribution later. Two consequences shape
the API:

- The diagram is allowed to be **incomplete** at all times. A node may declare
  parents that are not yet in the diagram (a dangling reference); a node may
  have no ``dist`` yet. These are legitimate intermediate states.
- **Inference consumes a validated snapshot, never the live workspace.**
  :meth:`InfluenceDiagram.validate` is the explicit gate: it collects every
  problem that would block inference (dangling refs, cycles, unconfigured or
  stale distributions) and returns them as structured
  :class:`DiagramProblem` instances. :meth:`InfluenceDiagram.snapshot` produces
  a point-in-time view only when validation passes.

Acyclicity is enforced *eagerly* on :meth:`add_edge` (a cycle is never a useful
intermediate state) and *defensively* in :meth:`validate` (a node can be
mutated directly through its own ``add_parent``, bypassing the diagram, so
:meth:`validate` is the robust backstop).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from decisionpy.graph.chance_node import ChanceNode, Consistency, DistFactory


class ProblemKind(Enum):
    """
    Category of a :class:`DiagramProblem`.

    :var DANGLING_PARENT: A node lists a parent that is not in the diagram.
    :var CYCLE: The graph contains a cycle.
    :var UNCONFIGURED: A node has no ``dist`` set; inference cannot run on it.
    :var STALE: A node's ``dist`` signature does not match its parents.
    """

    DANGLING_PARENT = "dangling_parent"
    CYCLE = "cycle"
    UNCONFIGURED = "unconfigured"
    STALE = "stale"


@dataclass(frozen=True)
class DiagramProblem:
    """
    A single issue found by :meth:`InfluenceDiagram.validate`.

    Carries enough structure for a UI or LLM to render or act on the problem
    without parsing prose.
    """

    kind: ProblemKind
    node: str
    message: str


@dataclass(frozen=True)
class Snapshot:
    """
    A point-in-time, validated view of a diagram — what inference consumes.

    Produced by :meth:`InfluenceDiagram.snapshot` only after validation passes.
    It holds references to the (still mutable) nodes, so it is a *logical*
    snapshot: editing the diagram after taking a snapshot invalidates it. A
    truly deep-frozen copy is deferred until the translator exists and pins
    down the shape it needs.
    """

    nodes: tuple[tuple[str, ChanceNode], ...]
    order: tuple[str, ...]


class InfluenceDiagram:
    """
    A mutable directed acyclic graph of nodes representing an influence diagram.

    Currently supports chance nodes only; decision and utility nodes will be
    added in later milestones. A Bayesian network is simply an
    :class:`InfluenceDiagram` containing only chance nodes.

    Nodes are registered by name. Unlike a build-once container, a node may be
    added before its parents exist, edges may be wired and unwired freely, and
    distributions may be set at any time. :meth:`validate` is the checkpoint
    that decides whether the current state is sound enough to infer on.
    """

    def __init__(self) -> None:
        """Initialize an empty diagram."""
        self._nodes: dict[str, ChanceNode] = {}

    # --- registration -------------------------------------------------------

    def add_node(self, node: ChanceNode) -> None:
        """
        Register a node in the diagram.

        The node's name must be unique. Its parents need *not* be present yet
        — dangling references are tolerated during construction and caught by
        :meth:`validate`.

        :param ChanceNode node: The node to add.
        :raises ValueError: If a node with this name is already registered.
        """
        if node.name in self._nodes:
            raise ValueError(f"A node named {node.name!r} is already in the diagram.")
        self._nodes[node.name] = node

    def remove_node(self, name: str) -> ChanceNode:
        """
        Remove ``name`` from the diagram and scrub it from every survivor's parents.

        Removing a node that other nodes still reference as a parent cleans
        those references up (via each survivor's ``remove_parent``), so no
        dangling reference to the removed name is left behind.

        :raises KeyError: If ``name`` is not in the diagram.
        """
        node = self._nodes.pop(name)
        for survivor in self._nodes.values():
            if name in survivor.parents:
                survivor.remove_parent(name)
        return node

    # --- edges --------------------------------------------------------------

    def add_edge(self, parent: str, child: str) -> None:
        """
        Wire an edge ``parent -> child``.

        Both endpoints must already be in the diagram (use :meth:`add_node`
        first). The child's parent set is updated through its own
        ``add_parent``, so the child's field validation runs — and its
        ``dist`` may become stale as a result, which is expected.

        Cycle prevention: if a path ``child -> ... -> parent`` already exists,
        adding ``parent -> child`` would close a cycle, and the call is
        rejected. Idempotent: a duplicate edge is a no-op.

        :raises KeyError: If either endpoint is not in the diagram.
        :raises ValueError: If the edge would create a cycle (including a
            self-loop).
        """
        if parent not in self._nodes:
            raise KeyError(f"Parent {parent!r} is not in the diagram.")
        if child not in self._nodes:
            raise KeyError(f"Child {child!r} is not in the diagram.")
        if parent == child:
            raise ValueError(f"Node {parent!r} cannot be its own parent.")
        if parent in self._nodes[child].parents:
            return  # idempotent
        if _reaches(self._children_of, start=child, target=parent):
            raise ValueError(
                f"Edge {parent!r} -> {child!r} would create a cycle "
                f"(a path {child!r} -> ... -> {parent!r} already exists)."
            )
        self._nodes[child].add_parent(parent)

    def remove_edge(self, parent: str, child: str) -> None:
        """
        Remove the edge ``parent -> child`` if present; no-op otherwise.

        Going through the child's ``remove_parent`` keeps its parent set and
        the (derived) adjacency consistent.
        """
        if child in self._nodes and parent in self._nodes[child].parents:
            self._nodes[child].remove_parent(parent)

    # --- distribution / state mutators --------------------------------------

    def set_dist(self, name: str, dist: DistFactory | None) -> None:
        """Set the distribution factory on node ``name`` (field-validated)."""
        self._nodes[name].dist = dist

    # --- topology -----------------------------------------------------------

    def topological_sort(self) -> tuple[str, ...]:
        """
        Return node names in a topological order (parents before children).

        Uses Kahn's algorithm over edges between *existing* nodes; dangling
        parent references are ignored for ordering. Raises if a cycle is
        present (possible only if a node was mutated directly, bypassing
        :meth:`add_edge`'s cycle check).
        """
        nodes = self._nodes
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
        if len(order) != len(nodes):
            raise ValueError(
                "Diagram has a cycle; cannot topologically sort. "
                "Run validate() to locate it."
            )
        return tuple(order)

    def parents_of(self, name: str) -> tuple[str, ...]:
        """Return the declared parents of ``name``."""
        return self._nodes[name].parents

    def children_of(self, name: str) -> tuple[str, ...]:
        """Return names of nodes that list ``name`` as a parent."""
        return tuple(
            other_name
            for other_name, other in self._nodes.items()
            if name in other.parents
        )

    # --- validation ---------------------------------------------------------

    def validate(self) -> list[DiagramProblem]:
        """
        Collect every problem that would block inference.

        Returns an empty list for a diagram that is sound to infer on:
        no dangling parent references, no cycles, and every node
        :attr:`~Consistency.CONSISTENT`. The list is preferred over raising
        on the first problem so a UI or LLM can surface all issues at once.
        """
        problems: list[DiagramProblem] = []

        # Dangling parents: a declared parent not present in the diagram.
        for name, node in self._nodes.items():
            for parent in node.parents:
                if parent not in self._nodes:
                    problems.append(
                        DiagramProblem(
                            kind=ProblemKind.DANGLING_PARENT,
                            node=name,
                            message=(
                                f"Parent {parent!r} of {name!r} is not in the diagram."
                            ),
                        )
                    )

        # Per-node consistency: inference needs a configured, matching dist.
        for name, node in self._nodes.items():
            state = node.consistency
            if state is Consistency.UNCONFIGURED:
                problems.append(
                    DiagramProblem(
                        kind=ProblemKind.UNCONFIGURED,
                        node=name,
                        message=f"Node {name!r} has no `dist` configured.",
                    )
                )
            elif state is Consistency.STALE:
                problems.append(
                    DiagramProblem(
                        kind=ProblemKind.STALE,
                        node=name,
                        message=(
                            f"Node {name!r}'s `dist` signature does not match "
                            f"its parents {node.parents!r}."
                        ),
                    )
                )

        # Defensive cycle check: a node may have been mutated directly via its
        # own `add_parent`, bypassing `add_edge`. Kahn's leftover-node count is
        # the signal.
        try:
            self.topological_sort()
        except ValueError:
            problems.append(
                DiagramProblem(
                    kind=ProblemKind.CYCLE,
                    node="*",
                    message="Diagram contains a cycle.",
                )
            )

        return problems

    def snapshot(self) -> Snapshot:
        """
        Return a validated, point-in-time view of the diagram.

        :raises ValueError: If :meth:`validate` reports any problem, with the
            problems listed in the message.
        """
        problems = self.validate()
        if problems:
            formatted = "; ".join(p.message for p in problems)
            raise ValueError(f"Diagram is not valid: {formatted}")
        order = self.topological_sort()
        nodes = tuple((name, self._nodes[name]) for name in order)
        return Snapshot(nodes=nodes, order=order)

    # --- lookups ------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        """Whether a node named ``name`` is registered."""
        return name in self._nodes

    def __getitem__(self, name: str) -> ChanceNode:
        """Return the node registered under ``name``; raises ``KeyError`` if absent."""
        return self._nodes[name]

    def __len__(self) -> int:
        """The number of nodes registered."""
        return len(self._nodes)

    def __iter__(self):
        """Iterate over node names (insertion order, not topological)."""
        return iter(self._nodes)

    @property
    def names(self) -> tuple[str, ...]:
        """All node names in insertion order."""
        return tuple(self._nodes)

    # --- internals ----------------------------------------------------------

    def _children_of(self, name: str) -> tuple[str, ...]:
        """Names of nodes that list ``name`` as a parent (derived on demand)."""
        return self.children_of(name)


def _reaches(children_of, start: str, target: str) -> bool:
    """
    Whether ``target`` is reachable from ``start`` following child edges.

    DFS over the (derived) child adjacency. Used by :meth:`add_edge` to detect
    whether a prospective edge would close a cycle.
    """
    stack = [start]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children_of(current))
    return False
