"""
Influence diagram — the graph container.

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
  `InfluenceDiagram.validate` is the explicit gate: it collects every
  problem that would block inference (dangling refs, cycles, unconfigured or
  stale distributions) and returns them as structured
  `DiagramProblem` instances. `InfluenceDiagram.snapshot` produces
  a point-in-time view only when validation passes.

Acyclicity is enforced *eagerly* on `InfluenceDiagram.add_edge` (a cycle is
never a useful intermediate state) and *defensively* in
`InfluenceDiagram.validate` (a node can be
mutated directly through its own ``add_parent``, bypassing the diagram, so
validation is the robust backstop).

Three node kinds live in one container: chance, decision, and utility (see
`NodeKind`). They share the ``parents`` field, so
topological ordering and cycle prevention work uniformly across them. Utility
nodes are *sinks* — `InfluenceDiagram.add_edge` rejects an edge giving one a
child (eager, alongside the cycle check), and `InfluenceDiagram.validate`
reports any utility node that
nonetheless has children (defensive backstop, the same two-layer treatment).
"""

from __future__ import annotations

from dataclasses import dataclass

from pylimid.graph.chance_node import ChanceNode, DistFactory
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.node import Consistency, Node
from pylimid.graph.render.graphviz import available as graphviz_available
from pylimid.graph.render.graphviz import to_svg
from pylimid.graph.render.mermaid import to_mermaid
from pylimid.graph.utility_node import UtilityNode
from pylimid.graph.validation import DiagramProblem, DiagramProblemKind


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


@dataclass(frozen=True)
class DiagramSnapshot:
    """
    A point-in-time, validated view of a diagram — what inference consumes.

    Produced by `InfluenceDiagram.snapshot` only after validation passes.
    It holds references to the (still mutable) nodes, so it is a *logical*
    snapshot: editing the diagram after taking a snapshot invalidates it.
    """

    nodes: tuple[tuple[str, Node], ...]

    @property
    def order(self) -> tuple[str, ...]:
        """Node names in topological order (derived from nodes)."""
        return tuple(name for name, _ in self.nodes)


class InfluenceDiagram:
    """
    A mutable directed acyclic graph of nodes representing an influence diagram.

    Currently supports chance, decision, and utility nodes. A Bayesian network
    is simply an `InfluenceDiagram` containing only chance nodes; the
    moment a decision or utility node is added it becomes an influence diagram.

    Nodes are registered by name. Unlike a build-once container, a node may be
    added before its parents exist, edges may be wired and unwired freely, and
    distributions may be set at any time. `validate` is the checkpoint
    that decides whether the current state is sound enough to infer on.
    """

    def __init__(self) -> None:
        """Initialize an empty diagram."""
        self._nodes: dict[str, Node] = {}

    # --- registration -------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """
        Register a node in the diagram.

        The node's name must be unique. Its parents need *not* be present yet
        — dangling references are tolerated during construction and caught by
        `validate`.

        Args:
            node: The node to add.

        Raises:
            ValueError: If a node with this name is already registered.
        """
        if node.name in self._nodes:
            raise ValueError(f"A node named {node.name!r} is already in the diagram.")
        self._nodes[node.name] = node

    def remove_node(self, name: str) -> Node:
        """
        Remove ``name`` from the diagram and scrub it from every survivor's parents.

        Removing a node that other nodes still reference as a parent cleans
        those references up (via each survivor's ``remove_parent``), so no
        dangling reference to the removed name is left behind.

        Raises:
            KeyError: If ``name`` is not in the diagram.
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

        Both endpoints must already be in the diagram (use `add_node`
        first). The child's parent set is updated through its own
        ``add_parent``, so the child's field validation runs — and its
        ``dist`` / ``values`` may become stale as a result, which is expected.

        Cycle prevention: if a path ``child -> ... -> parent`` already exists,
        adding ``parent -> child`` would close a cycle, and the call is
        rejected. Utility-sink prevention: an edge whose *parent* is a utility
        node is rejected (the parent would gain a child, and a payoff must be
        terminal). Both are eager checks (a cycle and a utility-with-child are
        never useful intermediate states). Idempotent: a duplicate edge is a
        no-op.

        Raises:
            KeyError: If either endpoint is not in the diagram.
            ValueError: If the edge would create a cycle (including a
                self-loop), or if the parent is a sink node (utility).
        """
        if parent not in self._nodes:
            raise KeyError(f"Parent {parent!r} is not in the diagram.")
        if child not in self._nodes:
            raise KeyError(f"Child {child!r} is not in the diagram.")
        if self._nodes[parent].is_sink:
            raise ValueError(
                f"Node {parent!r} is a utility node and cannot have children "
                f"(rejected edge {parent!r} -> {child!r})."
            )
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
        """
        Set the distribution factory on chance node ``name`` (field-validated).

        Raises:
            TypeError: If ``name`` is not a chance node. Decision nodes carry
                an action ``states`` instead, and utility nodes a ``values``
                function — set those directly on the node.
        """
        node = self._nodes[name]
        if not isinstance(node, ChanceNode):
            raise TypeError(
                f"`set_dist` applies to chance nodes only; {name!r} is a "
                f"{node.kind.value} node."
            )
        node.dist = dist

    # --- topology -----------------------------------------------------------

    def topological_sort(self) -> tuple[str, ...]:
        """
        Return node names in a topological order (parents before children).

        Uses Kahn's algorithm over edges between *existing* nodes; dangling
        parent references are ignored for ordering. Raises if a cycle is
        present (possible only if a node was mutated directly, bypassing
        the cycle check in `add_edge`).
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
        `Consistency.CONSISTENT`. The list is preferred over raising
        on the first problem so a UI or LLM can surface all issues at once.
        """
        problems: list[DiagramProblem] = []

        # Dangling parents: a declared parent not present in the diagram.
        for name, node in self._nodes.items():
            for parent in node.parents:
                if parent not in self._nodes:
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
        for name, node in self._nodes.items():
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

        # Defensive utility-sink check: `add_edge` rejects a utility child
        # eagerly, but a node can be mutated directly via its own `add_parent`
        # (bypassing the diagram), so a utility node may nonetheless appear as
        # some other node's parent. This is the backstop, mirroring the cycle
        # check below.
        for name, node in self._nodes.items():
            if node.is_sink and self.children_of(name):
                problems.append(
                    DiagramProblem(
                        kind=DiagramProblemKind.UTILITY_NOT_SINK,
                        node=name,
                        message=(
                            f"Utility node {name!r} must be terminal but has "
                            f"children {self.children_of(name)!r}."
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
                    kind=DiagramProblemKind.CYCLE,
                    node="*",
                    message="Diagram contains a cycle.",
                )
            )

        return problems

    def probe_discrete_parents(self) -> list[DiagramProblem]:
        """
        Check that every node's callable actually uses each discrete parent.

        For every chance or utility node, each discrete parent is varied
        across its states (the other parents held fixed) and the node's
        callable is evaluated per value. When the returned distribution
        never changes as a parent changes, the callable does not use that
        parent — the classic silent mistake, e.g. a probability table with
        fewer rows than the parent has states, where JAX indexing silently
        clamps out-of-range values back to the last row.

        Unlike `validate`, this executes the node callables (arbitrary
        user code), so it is an explicit, opt-in check — never run
        automatically by `snapshot`. Continuous parents (no declared
        ``states``) are skipped: there is nothing to enumerate.

        A finding is a warning, not an error: a deliberately independent
        node (its callable genuinely ignores a parent) looks identical and
        is a legitimate model. The check cannot detect callables that use
        the parent but map it to wrong values.

        Returns:
            One `DiagramProblem` per (node, discrete parent) pair
            whose callable output is insensitive to the parent.
        """
        problems: list[DiagramProblem] = []
        for name, node in self._nodes.items():
            if isinstance(node, ChanceNode):
                callable_ = node.dist
            elif isinstance(node, UtilityNode):
                callable_ = node.values
            else:
                continue
            if callable_ is None:
                continue  # UNCONFIGURED — validate() reports it
            for parent in node.parents:
                parent_node = self._nodes.get(parent)
                if not isinstance(parent_node, (ChanceNode, DecisionNode)):
                    continue  # dangling or continuous (no states): out of scope
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
                                f"Node {name!r}'s callable returns the same "
                                f"distribution for every value of parent "
                                f"{parent!r}; {parent} appears to have no "
                                f"effect. Warning: a deliberately independent "
                                f"node looks identical."
                            ),
                        )
                    )
        return problems

    def snapshot(self) -> DiagramSnapshot:
        """
        Return a validated, point-in-time `DiagramSnapshot` of the diagram.

        Raises:
            ValueError: If `validate` reports any problem, with the
                problems listed in the message.
        """
        problems = self.validate()
        if problems:
            formatted = "; ".join(p.message for p in problems)
            raise ValueError(f"Diagram is not valid: {formatted}")
        order = self.topological_sort()
        nodes = tuple((name, self._nodes[name]) for name in order)
        return DiagramSnapshot(nodes=nodes)

    # --- lookups ------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        """Whether a node named ``name`` is registered."""
        return name in self._nodes

    def __getitem__(self, name: str) -> Node:
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

    def __repr__(self) -> str:
        """A deterministic summary, used when Graphviz cannot draw a figure."""
        return f"InfluenceDiagram({', '.join(self.names)})"

    # --- rendering ----------------------------------------------------------

    def to_mermaid(self) -> str:
        """
        Render the diagram as a Mermaid ``flowchart`` source string.

        Deterministic (topological node order, declared parent order for
        edges) and usable on the live workspace: no validation required, so
        an incomplete diagram renders with dangling parents as dashed ghost
        nodes and non-consistent nodes tinted via ``classDef``. See
        `pylimid.graph.render.mermaid`.

        Returns:
            Mermaid flowchart source; paste into a Mermaid renderer to view.
        """
        return to_mermaid(self)

    def to_svg(self) -> str:
        """
        Render the diagram as an SVG string with Graphviz.

        Shells out to the Graphviz ``dot`` binary, which must be on ``PATH``.
        In Jupyter the diagram renders itself through this method; see
        `pylimid.graph.render.graphviz`.

        Returns:
            SVG source.

        Raises:
            FileNotFoundError: If the ``dot`` binary is not on ``PATH``.
            subprocess.CalledProcessError: If ``dot`` rejects the graph.
        """
        return to_svg(self)

    def _repr_svg_(self) -> str | None:
        """
        Jupyter display hook: the SVG rendering, or ``None`` without ``dot``.

        Returning ``None`` when the ``dot`` binary is missing makes Jupyter
        fall back to the plain repr instead of failing the display.
        """
        if not graphviz_available():
            return None
        return to_svg(self)

    # --- internals ----------------------------------------------------------

    def _children_of(self, name: str) -> tuple[str, ...]:
        """Names of nodes that list ``name`` as a parent (derived on demand)."""
        return self.children_of(name)


def _reaches(children_of, start: str, target: str) -> bool:
    """
    Whether ``target`` is reachable from ``start`` following child edges.

    DFS over the (derived) child adjacency. Used by
    `InfluenceDiagram.add_edge` to detect whether a prospective edge would
    close a cycle.
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
