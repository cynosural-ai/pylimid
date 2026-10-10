"""
Decision node — a variable the agent controls.

A decision node has **no distribution**: its value is chosen, not sampled. What
it does carry is:

- an **information set** (the ``parents`` field — the variables observed when
  the decision is made), and
- an **action space** (``states`` — the available choices).

The information set is structural, not causal
---------------------------------------------
A decision's ``parents`` are *not* a statistical dependency. They record what
the agent observes before acting. This is the standard influence-diagram
convention: the same ``parents`` tuple that drives topological ordering and
cycle detection for chance nodes is reused here, but its meaning is
"information available at decision time." Keeping one field name means the
container's shared bookkeeping (validation, ordering) is reused unchanged —
see `pylimid.graph.node`.

Consistency
-----------
A decision is `Consistency.CONSISTENT` once its action ``states`` are
declared; before that it is `Consistency.UNCONFIGURED`. There is no
`Consistency.STALE` state for a decision: the action space does not depend
on the information set's size, so adding or removing an information parent
never invalidates it.

Continuous decisions are not supported: ``states=None`` leaves the node
unconfigured.
"""

from dataclasses import dataclass, field
from typing import Any

from pylimid.graph.node import (
    Consistency,
    Node,
    NodeKind,
)


@dataclass
class DecisionNode(Node):
    """
    A decision variable in an influence diagram.

    Attributes:
        name: Identifier for the node.
        parents: The **information set** — names of the variables observed
            when this decision is made. Not a causal dependency. May be edited
            after construction.
        states: Labels for the available actions. ``None`` (the default) marks
            the decision as not-yet-configured. Values flowing through the
            graph are integer indices into this tuple.
    """

    states: tuple[str, ...] | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate ``states`` (decision-specific) then defer to base."""
        if key == "states":
            _validate_states(value)
        super().__setattr__(key, value)

    # --- kind / structure ---------------------------------------------------

    @property
    def kind(self) -> NodeKind:
        """`NodeKind.DECISION`."""
        return NodeKind.DECISION

    @property
    def is_discrete(self) -> bool:
        """
        Whether the decision has a declared, enumerable action space.

        ``True`` when ``states`` is set (a discrete decision, the kind the
        solvers handle). ``False`` when ``states`` is ``None``, which marks
        the decision as not-yet-configured.
        """
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> Consistency:
        if self.states is None:
            return Consistency.UNCONFIGURED
        return Consistency.CONSISTENT

    def consistency_message(self, state: Consistency) -> str:
        """Decision-node wording for a non-CONSISTENT state (see base)."""
        # UNCONFIGURED is the only non-CONSISTENT state for a decision.
        return (
            f"Decision {self.name!r} has no action space (`states`) declared; "
            f"set `states` to the available actions."
        )


# --- decision-specific field validators -------------------------------------


def _validate_states(states: Any) -> None:
    """Validate the action ``states`` (mirrors the chance-node states check)."""
    if states is None:
        return
    if not isinstance(states, tuple):
        raise TypeError(
            f"`states` must be a tuple or None, got {type(states).__name__}."
        )
    if not states:
        raise ValueError(
            "`states`, when provided, must be non-empty; use `None` for an "
            "undeclared (or future continuous) decision."
        )
    if not all(isinstance(s, str) and s.strip() for s in states):
        raise ValueError(f"`states` must all be non-empty strings, got {states!r}.")
    if len(set(states)) != len(states):
        raise ValueError(f"`states` must be unique, got {states!r}.")
