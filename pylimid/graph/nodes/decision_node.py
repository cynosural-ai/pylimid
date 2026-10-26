"""
Decision node — a variable the agent controls.

A decision node has **no chance distribution**: a policy controls its action. What
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
see `pylimid.graph.nodes.node`.

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

from pylimid.graph.nodes.node import (
    Consistency,
    Node,
    NodeKind,
)
from pylimid.policy import PolicyFamily


@dataclass
class DecisionNode(Node):
    """
    A decision variable in an influence diagram.

    A decision chooses among a finite set of actions. Tabular policies require every
    parent to declare named finite states; numerical counts and continuous parents
    cannot index these tables. Supplied deterministic functions and fitted
    `CategoricalPolicy` rules can use continuous observations during forward evaluation
    with `infer`. Declare a `PolicyFamily`, such as `LinearSoftmax` or
    `TabularSoftmax`, on every decision to fit their joint policy through `solve`.
    Its ``parents`` are the information available when it acts; earlier observations
    and decisions are remembered only when included explicitly.
    ``states=None`` leaves the decision unconfigured.

    Attributes:
        name: Identifier for the node.
        parents: The **information set** — names of the variables observed
            when this decision is made. Not a causal dependency. May be edited
            after construction.
        states: Labels for the available actions. ``None`` (the default) marks
            the decision as not-yet-configured. Values flowing through the
            graph are integer indices into this tuple.
        policy_family: Optional creator-selected `PolicyFamily`.
            `TabularSoftmax` requires scalar categorical information. Use
            `solve` to fit all declared rules jointly.
            ``None`` allows supplied rules for evaluation with `infer`; `solve` requires
            a declared family on every decision.
    """

    states: tuple[str, ...] | None = field(default=None, kw_only=True)
    policy_family: PolicyFamily | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate decision-specific fields, then defer to the base node."""
        if key == "states":
            _validate_states(value)
        elif (
            key == "policy_family"
            and value is not None
            and not isinstance(value, PolicyFamily)
        ):
            raise TypeError("`policy_family` must be a PolicyFamily instance or None.")
        super().__setattr__(key, value)

    # --- kind / structure ---------------------------------------------------

    @property
    def kind(self) -> NodeKind:
        """`NodeKind.DECISION`."""
        return NodeKind.DECISION

    @property
    def is_categorical(self) -> bool:
        """
        Whether the decision declares named actions through ``states``.

        ``False`` when ``states`` is ``None``, which marks the decision as
        not-yet-configured.
        """
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def consistency_message(self, state: Consistency) -> str:
        """
        Describe the missing action states.

        Args:
            state: The node's ``UNCONFIGURED`` consistency state.

        Returns:
            A description of the configuration problem.
        """
        # UNCONFIGURED is the only non-CONSISTENT state for a decision.
        return (
            f"Decision {self.name!r} has no action space (`states`) declared; "
            f"set `states` to the available actions."
        )

    def _compute_consistency(self) -> Consistency:
        if self.states is None:
            return Consistency.UNCONFIGURED
        return Consistency.CONSISTENT


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
