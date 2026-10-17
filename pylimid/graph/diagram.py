"""
Influence diagram — the graph container.

Mutable workspace, validated gate
---------------------------------
The diagram is a workspace that an external author — a script, a UI, or an LLM
driving the model over a tool interface — edits incrementally: drop a node
here, wire an arc there, fill in a distribution later. Two consequences shape
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

Structural soundness is checked in one place. Editing methods only refuse
requests they cannot carry out (a duplicate name, an arc into a node that is
not in the diagram, a malformed field); cycles and utility nodes with children
are reported by `InfluenceDiagram.validate`, however the diagram was built.

Three node kinds live in one container: chance, decision, and utility (see
`NodeKind`). They share the ``parents`` field, so topological ordering and
cycle detection work uniformly across them. Utility nodes are *sinks*: a
payoff with a child is reported as a problem.
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
    A diagram's nodes in topological order, as used by inference and solvers.

    Obtain a validated snapshot with `InfluenceDiagram.snapshot`. The snapshot holds
    references to the original nodes, rather than copies. Take a new snapshot after
    editing the diagram or its nodes.

    Attributes:
        nodes: ``(name, node)`` pairs in topological order.
    """

    nodes: tuple[tuple[str, Node], ...]

    @property
    def order(self) -> tuple[str, ...]:
        """Node names in topological order (derived from nodes)."""
        return tuple(name for name, _ in self.nodes)


class InfluenceDiagram:
    """
    A mutable workspace for building an influence diagram.

    Currently supports chance, decision, and utility nodes. A Bayesian network
    is simply an `InfluenceDiagram` containing only chance nodes; the
    moment a decision or utility node is added it becomes an influence diagram.

    Nodes are registered by name. A node may be added before its parents exist, and
    distributions and utility functions may be configured later. `validate` checks the
    diagram's structure and configuration. `infer` and `solve` run this check
    automatically before using the diagram.
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
        Remove a node and its references from the remaining nodes' parents.

        Functions on the affected nodes may need updating to match their new parents.

        Args:
            name: Name of the node to remove.

        Returns:
            The removed node.

        Raises:
            KeyError: If ``name`` is not in the diagram.
        """
        node = self._nodes.pop(name)
        for survivor in self._nodes.values():
            if name in survivor.parents:
                survivor.remove_parent(name)
        return node

    # --- arcs --------------------------------------------------------------

    def add_arc(self, parent: str, child: str) -> None:
        """
        Wire an arc ``parent -> child``.

        The child must already be in the diagram. The parent may be added later. Adding
        a parent may require updating the child's distribution or utility function to
        accept it.

        Structural problems (a cycle, a utility node with a child) are not
        rejected here; `validate` reports them, as it does for arcs declared
        in `add_node`. Adding an existing arc has no effect.

        Args:
            parent: Name of the parent node.
            child: Name of the child node.

        Raises:
            KeyError: If the child is not in the diagram.
            ValueError: If the arc is a self-loop.
        """
        if child not in self._nodes:
            raise KeyError(f"Child {child!r} is not in the diagram.")
        self._nodes[child].add_parent(parent)

    def remove_arc(self, parent: str, child: str) -> None:
        """
        Remove the arc ``parent -> child`` if present; no-op otherwise.

        Removing a parent may require updating the child's distribution or utility
        function. A missing child or arc has no effect.

        Args:
            parent: Name of the parent node.
            child: Name of the child node.
        """
        if child in self._nodes and parent in self._nodes[child].parents:
            self._nodes[child].remove_parent(parent)

    # --- distribution / state mutators --------------------------------------

    def set_dist(self, name: str, dist: DistFactory | None) -> None:
        """
        Set the distribution function on a chance node.

        Args:
            name: Name of the chance node.
            dist: Callable returning a NumPyro distribution from parent values, or
                ``None`` to leave the node unconfigured.

        Raises:
            KeyError: If ``name`` is not in the diagram.
            TypeError: If the node is not a chance node, or ``dist`` is neither callable
                nor ``None``. Set a decision's ``states`` or a utility's ``values``
                directly on that node.
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

        Missing parent references are ignored for ordering; `validate` reports them
        separately.

        Returns:
            Node names with parents before children.

        Raises:
            ValueError: If the diagram contains a cycle.
        """
        order = self._kahn_order()
        if len(order) != len(self._nodes):
            raise ValueError(
                "Diagram has a cycle; cannot topologically sort. "
                "Run validate() to locate it."
            )
        return tuple(order)

    def _kahn_order(self) -> list[str]:
        """
        Kahn's algorithm, stopping where a cycle blocks it.

        Returns every node when the diagram is acyclic. Otherwise the nodes
        left out are those on a cycle or downstream of one.
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
        return order

    def _find_cycle(self, blocked: set[str]) -> tuple[str, ...]:
        """
        One cycle among the nodes Kahn's algorithm could not order.

        Every blocked node has a blocked parent, so walking up parents from
        any of them must eventually repeat a node. Returned in arc order
        (parent before child), starting from the repeated node.
        """
        current = min(blocked)
        path: list[str] = []
        position: dict[str, int] = {}
        while current not in position:
            position[current] = len(path)
            path.append(current)
            current = next(p for p in self._nodes[current].parents if p in blocked)
        return tuple(reversed(path[position[current] :]))

    def parents_of(self, name: str) -> tuple[str, ...]:
        """
        Return the declared parents of a node.

        Args:
            name: Name of the node.

        Returns:
            Parent names in their declared order, including any missing from the
            diagram.

        Raises:
            KeyError: If ``name`` is not in the diagram.
        """
        return self._nodes[name].parents

    def children_of(self, name: str) -> tuple[str, ...]:
        """
        Return the nodes that list a name as a parent.

        Args:
            name: Parent name to look up. It need not be in the diagram.

        Returns:
            Child names in insertion order, or an empty tuple if none refer to ``name``.
        """
        return tuple(
            other_name
            for other_name, other in self._nodes.items()
            if name in other.parents
        )

    def states_of(self, name: str) -> tuple[str, ...]:
        """
        Return the state labels of a discrete chance or decision node.

        Args:
            name: Name of the node.

        Returns:
            The node's ``states`` labels.

        Raises:
            KeyError: If ``name`` is not in the diagram.
            ValueError: If the node has no state labels — a continuous chance
                node, an unconfigured decision, or a utility node.
        """
        node = self._nodes[name]
        if not isinstance(node, (ChanceNode, DecisionNode)) or node.states is None:
            raise ValueError(
                f"Node {name!r} has no state labels; only discrete chance "
                f"and decision nodes do."
            )
        return node.states

    # --- validation ---------------------------------------------------------

    def validate(self) -> list[DiagramProblem]:
        """
        Check the diagram's structure and node configuration.

        Checks for missing parents, cycles, unconfigured nodes, callable signatures that
        do not match their parents, and utility nodes with children. It does not
        evaluate distribution or utility functions, check their numerical outputs or
        test whether the diagram is solvable by backward induction.

        `infer` and `solve` run this check automatically. Use `probe_discrete_parents`
        separately to look for parents whose effects may be missing.

        Returns:
            The detected problems, or an empty list if these checks pass.
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

        # A utility node is a payoff, so it must be terminal.
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

        blocked = set(self._nodes) - set(self._kahn_order())
        if blocked:
            cycle = self._find_cycle(blocked)
            path = " -> ".join(repr(name) for name in (*cycle, cycle[0]))
            problems.append(
                DiagramProblem(
                    kind=DiagramProblemKind.CYCLE,
                    node=cycle[0],
                    message=f"Diagram contains a cycle: {path}.",
                )
            )

        return problems

    def probe_discrete_parents(self) -> list[DiagramProblem]:
        """
        Look for discrete parents that have no effect at the tested inputs.

        For each configured chance or utility function, vary each discrete parent across
        its state indices while setting all other parents to zero, including continuous
        parents. Compare the returned distribution parameters or scalar utility values.
        This can expose mistakes such as a lookup table with only one row for a parent
        with several states.

        The check executes the functions and propagates any errors they raise. It runs
        only when called, never automatically through `validate` or `snapshot`.
        Continuous parents are not varied.

        Findings are warnings for inspection and do not block inference. A parent may be
        intentionally ignored or have an effect only at other inputs. The check does not
        establish that the model's numerical values are correct.

        Returns:
            One `DiagramProblem` per node and discrete parent whose output is unchanged
            at the tested inputs. Unconfigured functions are skipped.
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

        Returns:
            A topologically ordered view that shares the diagram's node objects. Take a
            new snapshot after editing the diagram or its nodes.

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

        No validation is required. Missing parents appear as dashed placeholders, and
        unconfigured or stale nodes are highlighted. Nodes appear in topological order,
        falling back to insertion order if the diagram contains a cycle.

        Returns:
            Mermaid flowchart source; paste into a Mermaid renderer to view.
        """
        return to_mermaid(self)

    def to_svg(self) -> str:
        """
        Render the diagram as an SVG string with Graphviz.

        Requires the Graphviz ``dot`` program on ``PATH``. No validation is required, so
        incomplete diagrams can also be drawn. In Jupyter, displaying the diagram uses
        this rendering automatically when ``dot`` is available.

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
