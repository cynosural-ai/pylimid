"""
Decision node — a variable the agent controls.

A decision node has **no chance distribution**: a policy controls its action. What
it does carry is:

- an **information set** (the ``parents`` field — the variables observed when
  the decision is made), and
- an **action space** (``states`` for categorical actions, or ``action_space``
  for a scalar continuous domain).

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
A decision is `Consistency.CONSISTENT` once either categorical action ``states``
or a continuous ``action_space`` is declared. With neither it is
`Consistency.UNCONFIGURED`; with both it is `Consistency.STALE`. Adding or
removing an information parent never invalidates the action declaration.

Continuous domains can be declared and validated, but execution through `infer`
and `solve` is not yet supported.
"""

from dataclasses import dataclass, field
from typing import Any

from pylimid.graph.action_space import Interval, Real
from pylimid.graph.nodes._validation import validate_states
from pylimid.graph.nodes.node import (
    Consistency,
    Node,
    NodeKind,
)
from pylimid.policy import PolicyFamily

__all__ = ["DecisionNode"]


@dataclass
class DecisionNode(Node):
    """
    A decision variable in an influence diagram.

    Declare categorical actions through ``states``, or a scalar continuous domain
    through ``action_space``. Setting both is a configuration conflict reported by
    `validate`. Continuous domains currently support declaration and graph validation
    only; `infer` and `solve` do not execute continuous decisions yet.

    For categorical decisions, tabular policies require every parent to declare named
    finite states. Supplied deterministic functions can use continuous observations
    during forward evaluation with `infer`. Fitted `CategoricalPolicy` rules also
    support conditioning on chance evidence with continuous or mixed information.
    Declare a `PolicyFamily`, such as `LinearSoftmax` or `TabularSoftmax`, on every
    categorical decision to fit their joint policy through `solve`.
    Its ``parents`` are the information available when it acts; earlier observations
    and decisions are remembered only when included explicitly.
    With neither ``states`` nor ``action_space``, the decision is unconfigured.

    Attributes:
        name: Identifier for the node.
        parents: The **information set** — names of the variables observed
            when this decision is made. Not a causal dependency. May be edited
            after construction.
        states: Labels for categorical actions. Values flowing through the graph
            are integer indices into this tuple. Leave ``None`` for a continuous
            decision or an unconfigured node.
        action_space: A `Real` or `Interval` domain for a scalar continuous action.
            Leave ``None`` when using categorical ``states``. A domain declaration
            does not select a distribution or policy family.
        policy_family: Optional creator-selected `PolicyFamily`.
            `TabularSoftmax` requires scalar categorical information. Use
            `solve` to fit all declared rules jointly.
            ``None`` allows supplied rules for evaluation with `infer`; `solve` requires
            a declared family on every decision.
    """

    states: tuple[str, ...] | None = field(default=None, kw_only=True)
    action_space: Real | Interval | None = field(default=None, kw_only=True)
    policy_family: PolicyFamily | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Validate decision-specific fields, then defer to the base node."""
        if key == "states":
            validate_states(value)
        elif key == "action_space" and value is not None:
            if not isinstance(value, (Real, Interval)):
                raise TypeError(
                    "`action_space` must be a Real or Interval instance or None."
                )
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

        ``False`` when ``states`` is ``None``. This can describe a declared
        continuous domain or an unconfigured decision.
        """
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def consistency_message(self, state: Consistency) -> str:
        """
        Describe a missing or conflicting action-space declaration.

        Args:
            state: The node's ``UNCONFIGURED`` or ``STALE`` consistency state.

        Returns:
            A description of the configuration problem.
        """
        if state is Consistency.STALE:
            return (
                f"Decision {self.name!r} declares both `states` and `action_space`; "
                "declare categorical states or a continuous action space, not both."
            )
        return (
            f"Decision {self.name!r} has no action space declared; "
            "set `states` for categorical actions or `action_space` "
            "to a Real or Interval domain."
        )

    def _compute_consistency(self) -> Consistency:
        if self.states is not None and self.action_space is not None:
            return Consistency.STALE
        if self.states is None and self.action_space is None:
            return Consistency.UNCONFIGURED
        return Consistency.CONSISTENT
