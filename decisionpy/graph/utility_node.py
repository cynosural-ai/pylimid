"""
Utility node — a deterministic payoff.

Implements the node portion of the utility representation; the mutability and
consistency model it inherits is in ``docs/diagram.md``, and the solving
strategy that consumes utility is in ``docs/decision_node.md``. See
``docs/utility_node.md`` for the dedicated design note.

A utility node is ``U(parents)`` — a deterministic scalar function of its
parents (chance and / or decision variables). It has **no distribution** and
**no states**: it is never sampled and carries no outcomes. Its value is
computed by the ``values`` callable, invoked as
``values(**parent_values) -> float``, keyed by parent name (same convention as
a chance node's ``dist``).

Utility nodes are terminal
--------------------------
In an influence diagram a payoff is always a **sink**: it has parents but no
children. The diagram enforces this on :meth:`add_edge
<decisionpy.graph.diagram.InfluenceDiagram.add_edge>` (rejecting an edge that
would give a utility node a child) and again in :meth:`validate
<decisionpy.graph.diagram.InfluenceDiagram.validate>` (the defensive backstop
against a node mutated directly through its own ``add_parent``). This mirrors
the eager-plus-defensive treatment of acyclicity — a utility-with-child is
never a useful intermediate state.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from decisionpy.graph.node import (
    Consistency as _Consistency,
)
from decisionpy.graph.node import (
    Node,
    NodeKind,
    _signature_matches,
)

#: Factory returning a utility value given resolved parent values. Invoked as
#: ``values(**parent_values)``.
ValueFactory = Callable[..., float]


@dataclass
class UtilityNode(Node):
    """
    A utility (payoff) node in an influence diagram.

    Attributes:
        name: Identifier for the node.
        parents: Names of the variables the payoff depends on. May be edited
            after construction.
        values: Callable invoked as ``values(**parent_values)`` returning a
            ``float`` payoff. ``None`` (the default) marks the node as
            not-yet-configured.
    """

    values: ValueFactory | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate ``values`` (utility-specific) then defer to base."""
        if key == "values":
            _validate_values(value)
        super().__setattr__(key, value)

    # --- kind / structure ---------------------------------------------------

    @property
    def kind(self) -> NodeKind:
        """``NodeKind.UTILITY``."""
        return NodeKind.UTILITY

    @property
    def is_sink(self) -> bool:
        """``True`` — a payoff is terminal: it may not have children."""
        return True

    @property
    def is_discrete(self) -> bool:
        """
        ``False`` — a utility node is a deterministic scalar, never sampled.

        This also routes any diagram containing a utility node away from
        variable elimination (which is discrete-BN only) toward a solver — the
        correct forward behavior, since a diagram with utilities is an
        influence diagram, not a Bayesian network.
        """
        return False

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> _Consistency:
        if self.values is None:
            return _Consistency.UNCONFIGURED
        if _signature_matches(self.values, self.parents):
            return _Consistency.CONSISTENT
        return _Consistency.STALE

    def consistency_message(self, state: _Consistency) -> str:
        """Utility-node wording for a non-CONSISTENT state (see base)."""
        if state is _Consistency.UNCONFIGURED:
            return f"Utility {self.name!r} has no `values` function configured."
        # STALE is the only remaining non-CONSISTENT state.
        return (
            f"Utility {self.name!r}'s `values` signature does not match its "
            f"parents {self.parents!r}; reconfigure `values` after changing "
            f"the parent set."
        )


# --- utility-specific field validators --------------------------------------


def _validate_values(values: Any) -> None:
    if values is not None and not callable(values):
        raise TypeError(
            f"`values` must be callable or None, "
            f"got object of type {type(values).__name__}."
        )
