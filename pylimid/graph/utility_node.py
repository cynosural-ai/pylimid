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
children. `InfluenceDiagram.validate` reports a utility node that has
children, like it reports a cycle, however the arc was created.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pylimid.graph.node import (
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

    The utility function receives parent values as keyword arguments and returns a
    scalar payoff. Categorical parents pass integer state indices; numerical parents,
    including counts, pass their actual values. The function must be JAX-traceable.
    When a diagram has several utility nodes, `solve` maximises their sum.

    A utility node cannot have children; `InfluenceDiagram.validate` reports any such
    arcs.

    Attributes:
        name: Identifier for the node.
        parents: Names of the variables the payoff depends on. May be edited
            after construction.
        values: Callable invoked as ``values(**parent_values)`` returning a scalar
            payoff. ``None`` (the default) marks the node as not-yet-configured.
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
        """`NodeKind.UTILITY`."""
        return NodeKind.UTILITY

    @property
    def is_sink(self) -> bool:
        """``True`` — a payoff is terminal: it may not have children."""
        return True

    @property
    def is_categorical(self) -> bool:
        """``False`` — a utility is a deterministic payoff with no state labels."""
        return False

    # --- consistency --------------------------------------------------------

    def consistency_message(self, state: Consistency) -> str:
        """
        Describe a missing utility function or a mismatch with the parents.

        Args:
            state: The node's ``UNCONFIGURED`` or ``STALE`` consistency state.

        Returns:
            A description of the configuration problem.
        """
        if state is Consistency.UNCONFIGURED:
            return f"Utility {self.name!r} has no `values` function configured."
        # STALE is the only remaining non-CONSISTENT state.
        return (
            f"The `values` signature for utility {self.name!r} does not match its "
            f"parents {self.parents!r}; reconfigure `values` after changing "
            f"the parent set."
        )

    def _compute_consistency(self) -> Consistency:
        if self.values is None:
            return Consistency.UNCONFIGURED
        if _signature_matches(self.values, self.parents):
            return Consistency.CONSISTENT
        return Consistency.STALE


# --- utility-specific field validators --------------------------------------


def _validate_values(values: Any) -> None:
    if values is not None and not callable(values):
        raise TypeError(
            f"`values` must be callable or None, "
            f"got object of type {type(values).__name__}."
        )
