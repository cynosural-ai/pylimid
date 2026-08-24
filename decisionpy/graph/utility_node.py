"""
Utility node — a deterministic payoff.

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
    Consistency,
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
        ``False`` — a utility node is not a random variable.

        It is a deterministic scalar payoff with no sampling domain, so
        ``is_discrete`` is vacuous for it. Engines must classify nodes by
        ``kind`` — a diagram with a utility node is an influence diagram, not
        a Bayesian network — never by ``is_discrete``.
        """
        return False

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> Consistency:
        if self.values is None:
            return Consistency.UNCONFIGURED
        if _signature_matches(self.values, self.parents):
            return Consistency.CONSISTENT
        return Consistency.STALE

    def consistency_message(self, state: Consistency) -> str:
        """Utility-node wording for a non-CONSISTENT state (see base)."""
        if state is Consistency.UNCONFIGURED:
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
