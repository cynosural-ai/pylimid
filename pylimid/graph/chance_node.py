"""
Chance node — a random variable conditioned on its parents.

A chance node is ``P(name | parents)`` — a random variable whose distribution
depends on its parents (a causal / statistical dependency, as distinct from a
decision node's information set). It inherits the shared mutable-node contract
from `Node` — field-level validation on every assignment, and the `Consistency`
gate (optional ``dist``, derived consistency) for cross-field checking.

The ``dist`` callable receives resolved parent values as **keyword arguments**
(keyed by parent name), so it is tied to parent *names*, which are stable,
rather than parent *order*, which is incidental.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pylimid.graph.node import Consistency, Node, NodeKind, _signature_matches

#: Factory returning a distribution object given resolved parent values.
#:
#: The graph layer is opaque to the concrete distribution type a backend uses,
#: so the return is typed ``Any`` here. The backend's translator narrows it.
DistFactory = Callable[..., Any]


@dataclass
class ChanceNode(Node):
    """
    A random variable in an influence diagram: ``P(name | parents)``.

    The node is mutable. Basic field checks, such as whether ``dist`` is callable and
    ``states`` contains unique labels, run when fields are set. The relationship between
    ``dist`` and ``parents`` is checked separately through `Node.consistency` or
    `InfluenceDiagram.validate`.

    The distribution function receives parent values as keyword arguments. Discrete
    parents pass integer state indices; continuous parents pass their numerical values.
    The function must be JAX-traceable and return a NumPyro distribution.

    With ``states``, the distribution must be Categorical with one label per category,
    or Bernoulli with exactly two labels. Both probability and logit parameterizations
    are supported, and zero-probability outcomes still need labels.
    ``states=None`` requires continuous support. Inference and solving check this
    contract when constructing the distribution, including for observed nodes;
    `InfluenceDiagram.validate` checks only the diagram's structure and declarations.

    Attributes:
        name: Identifier for the node.
        parents: Names of the nodes this one conditionally depends on. Empty
            for a root node. May be edited after construction.
        dist: Callable invoked as ``dist(**parent_values)`` returning the
            node's distribution. ``None`` (the default) marks the node as
            not-yet-configured.
        states: Unique, non-empty labels for the outcomes of a Categorical or Bernoulli
            distribution. Draws are integer indices into this tuple. ``None`` requires
            a distribution with continuous support.
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
        """`NodeKind.CHANCE`."""
        return NodeKind.CHANCE

    @property
    def is_categorical(self) -> bool:
        """
        Whether the node declares named categories through ``states``.

        This check does not call ``dist``. A node without categories may be
        unconfigured; its distribution's discreteness is known only when constructed.
        """
        return self.states is not None

    # --- consistency --------------------------------------------------------

    def _compute_consistency(self) -> Consistency:
        if self.dist is None:
            return Consistency.UNCONFIGURED
        if _signature_matches(self.dist, self.parents):
            return Consistency.CONSISTENT
        return Consistency.STALE

    def consistency_message(self, state: Consistency) -> str:
        """
        Describe a missing distribution or a mismatch with the parents.

        Args:
            state: The node's ``UNCONFIGURED`` or ``STALE`` consistency state.

        Returns:
            A description of the configuration problem.
        """
        if state is Consistency.UNCONFIGURED:
            return f"Node {self.name!r} has no `dist` configured yet."
        # STALE is the only remaining non-CONSISTENT state.
        return (
            f"The `dist` signature for node {self.name!r} does not match its "
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
