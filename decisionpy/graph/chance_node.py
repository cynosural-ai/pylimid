"""
Chance node — a random variable conditioned on its parents.

Implements the mutable node design settled in ``docs/chance_node.md`` and
``docs/diagram.md``; see ``docs/17_07_2026_mutability_design_decision.md`` for
the decision that makes this the library's sole node container model.

A chance node is ``P(name | parents)`` — a random variable whose distribution
depends on its parents (a causal / statistical dependency, as distinct from a
decision node's information set). It inherits the shared mutable-node contract
from :class:`~decisionpy.graph.node.Node`: field-level validation on every
assignment, and the :attr:`~decisionpy.graph.node.Consistency` gate (optional
``dist``, derived consistency) for cross-field checking.

The ``dist`` callable receives resolved parent values as **keyword arguments**
(keyed by parent name), so it is tied to parent *names*, which are stable,
rather than parent *order*, which is incidental.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from decisionpy.graph.node import Consistency as _Consistency
from decisionpy.graph.node import Node, NodeKind, _signature_matches

#: Factory returning a distribution object given resolved parent values.
#:
#: The graph layer is opaque to the concrete distribution type a backend uses,
#: so the return is typed ``Any`` here. The backend's translator narrows it.
DistFactory = Callable[..., Any]


@dataclass
class ChanceNode(Node):
    """
    A random variable in an influence diagram: ``P(name | parents)``.

    The node is mutable. Field-level validation runs on every assignment
    (including during construction), so a bad value is rejected at the moment
    it is set rather than later. Cross-field consistency (``dist`` signature
    vs ``parents``) is *not* enforced on assignment — it is queryable via
    :attr:`~decisionpy.graph.node.Node.consistency` and gated via
    :meth:`~decisionpy.graph.node.Node.validate`.

    Attributes:
        name: Identifier for the node.
        parents: Names of the nodes this one conditionally depends on. Empty
            for a root node. May be edited after construction.
        dist: Callable invoked as ``dist(**parent_values)`` returning the
            node's distribution. ``None`` (the default) marks the node as
            not-yet-configured.
        states: Human-facing labels for a discrete node's outcomes. ``None``
            marks the node as continuous. Values flowing through the graph are
            integer indices into this tuple.
    """

    dist: DistFactory | None = field(default=None, kw_only=True)
    states: tuple[str, ...] | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate ``dist`` / ``states`` (chance-specific) then defer to base."""
        if key == "dist":
            _validate_dist(value)
        elif key == "states":
            _validate_states(value)
        super().__setattr__(key, value)

    # --- kind / structure ---------------------------------------------------

    @property
    def kind(self) -> NodeKind:
        """``NodeKind.CHANCE``."""
        return NodeKind.CHANCE

    @property
    def is_discrete(self) -> bool:
        """Whether the node has a declared set of discrete states."""
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> _Consistency:
        if self.dist is None:
            return _Consistency.UNCONFIGURED
        if _signature_matches(self.dist, self.parents):
            return _Consistency.CONSISTENT
        return _Consistency.STALE

    def consistency_message(self, state: _Consistency) -> str:
        """Chance-node wording for a non-CONSISTENT state (see base)."""
        if state is _Consistency.UNCONFIGURED:
            return f"Node {self.name!r} has no `dist` configured yet."
        # STALE is the only remaining non-CONSISTENT state.
        return (
            f"Node {self.name!r}'s `dist` signature does not match its "
            f"parents {self.parents!r}; reconfigure `dist` after changing "
            f"the parent set."
        )


# --- chance-specific field validators ---------------------------------------


def _validate_dist(dist: Any) -> None:
    if dist is not None and not callable(dist):
        raise TypeError(
            f"`dist` must be callable or None, "
            f"got object of type {type(dist).__name__}."
        )


def _validate_states(states: Any) -> None:
    if states is None:
        return
    if not isinstance(states, tuple):
        raise TypeError(
            f"`states` must be a tuple or None, got {type(states).__name__}."
        )
    if not states:
        raise ValueError(
            "`states`, when provided, must be non-empty; use `None` for a "
            "continuous node."
        )
    if not all(isinstance(s, str) and s.strip() for s in states):
        raise ValueError(f"`states` must all be non-empty strings, got {states!r}.")
    if len(set(states)) != len(states):
        raise ValueError(f"`states` must be unique, got {states!r}.")
