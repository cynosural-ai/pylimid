"""
Decision node — a variable the agent controls.

Implements the node portion of the solving strategy settled in
``docs/decision_node.md``; the mutability and consistency model it inherits is
in ``docs/diagram.md``. The *solver* (intervention-scan, Strategy B for v0
discrete decisions; policy-as-parameters, Strategy A for continuous) is the
**next** milestone — this module defines only the node's representation in the
graph.

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
cycle prevention for chance nodes is reused here, but its meaning is
"information available at decision time." Keeping one field name means the
container's shared bookkeeping (validation, ordering) is reused unchanged —
see :mod:`decisionpy.graph.node`.

Consistency
-----------
A decision is :attr:`~decisionpy.graph.node.Consistency.CONSISTENT` once its
action ``states`` are declared; before that it is
:attr:`~decisionpy.graph.node.Consistency.UNCONFIGURED`. There is no STALE
state for a decision: the action space does not depend on the information set's
size, so adding or removing an information parent never invalidates it.

Continuous decisions (the v0 roadmap item that requires Strategy A) are
represented by ``states=None``; the node is then UNCONFIGURED in the graph
layer (the solver, not the graph, owns the continuous policy).
"""

from dataclasses import dataclass, field
from typing import Any

from decisionpy.graph.node import (
    Consistency as _Consistency,
)
from decisionpy.graph.node import (
    Node,
    NodeKind,
)


@dataclass
class DecisionNode(Node):
    """
    A decision variable in an influence diagram.

    :param str name: Identifier for the node.
    :param tuple[str, ...] parents: The **information set** — names of the
        variables observed when this decision is made. Not a causal
        dependency. May be edited after construction.
    :param tuple[str, ...] | None states: Labels for the available actions.
        ``None`` (the default) marks the decision as not-yet-configured (or,
        forward-looking, as continuous — to be owned by the solver). For a
        discrete decision, values flowing through the graph are integer
        indices into this tuple.
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
        """``NodeKind.DECISION``."""
        return NodeKind.DECISION

    @property
    def is_discrete(self) -> bool:
        """
        Whether the decision has a declared, enumerable action space.

        ``True`` when ``states`` is set (a discrete decision — what the v0
        intervention-scan solver handles). ``False`` when ``states`` is
        ``None``, which marks the decision as not-yet-configured or, in the
        future, continuous (Strategy A territory).
        """
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> _Consistency:
        if self.states is None:
            return _Consistency.UNCONFIGURED
        return _Consistency.CONSISTENT

    def consistency_message(self, state: _Consistency) -> str:
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
