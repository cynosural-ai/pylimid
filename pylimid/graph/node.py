"""
Node base — the shared contract for every node in an influence diagram.

The mutable-node contract is captured once here, on the base, so that
`ChanceNode`, `DecisionNode`, and `UtilityNode` inherit it uniformly rather than
re-implementing it per type.

Mutable by design
-----------------
A node is a workspace value that may be edited incrementally — add a parent,
set the distribution / action space / utility function, swap states — by an
external author: a script, a UI, or an LLM driving the diagram over a tool
interface. Because of that, a node is allowed to exist in an *inconsistent*
state: its configurable field (``dist`` / ``values`` / action ``states``) may
be unset, or its signature may not yet match its parents after an edge change.

Two layers of checking reflect this:

- **Field-level validation** runs on every assignment (including during
  construction). A bad value — empty name, duplicate parent, non-callable dist
  — is rejected the moment it is set.
- **Cross-field consistency** (the configurable field's signature vs
  ``parents``) is *not* enforced on assignment. It is queryable via
  `Node.consistency` and gated explicitly via `Node.validate`,
  which is the checkpoint inference runs against. Editing pauses wherever it
  likes; inference requires a ``CONSISTENT`` node.

The ``parents`` field
---------------------
Every node type carries a ``parents`` tuple of node names. Its semantics differ
by type, but the structural bookkeeping (validation, topological ordering,
cycle prevention) is identical, which is why it lives on the base:

- **Chance node** — the variables ``P(name | parents)`` conditions on (causal /
  statistical dependency).
- **Decision node** — the *information set*: the variables observed when the
  decision is made. Not a causal dependency.
- **Utility node** — the variables the payoff depends on.

Parent values are resolved and passed to the node's configurable callable as
**keyword arguments** keyed by parent name, so callables are tied to parent
*names* (stable) rather than parent *order* (incidental).
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(Enum):
    """
    Category of a node — the structural tag inference / solve dispatch reads.

    A diagram's node kinds decide what can be asked of it. A pure
    chance-node diagram is a Bayesian network, queried with `infer`; the
    moment a decision or utility node appears, it becomes an influence
    diagram, solved with `solve`.

    Attributes:
        CHANCE: A random variable — ``P(name | parents)``.
        DECISION: A variable the agent controls; ``states`` are its actions.
        UTILITY: A deterministic payoff — ``U(parents)``.
    """

    CHANCE = "chance"
    DECISION = "decision"
    UTILITY = "utility"


class Consistency(Enum):
    """
    State of the ``parents`` / configurable-field relationship on a node.

    Attributes:
        UNCONFIGURED: The node's configurable field (``dist`` for chance,
            ``values`` for utility, action ``states`` for decision) is unset —
            structure exists, the specification has not yet been supplied.
        STALE: The configurable field is set but its signature does not match
            ``parents`` — typically because a parent was added or removed since
            it was last configured.
        CONSISTENT: The configurable field is set and accepts ``parents`` as
            keyword arguments — the node is ready to be translated / inferred /
            solved.
    """

    UNCONFIGURED = "unconfigured"
    STALE = "stale"
    CONSISTENT = "consistent"


@dataclass
class Node:
    """
    Base class for every node in an influence diagram.

    Owns the shared mutable-node contract: a ``name`` and a ``parents`` tuple,
    field-level validation on every assignment, and the UNCONFIGURED / STALE /
    CONSISTENT consistency gate that inference consumes.

    This class is **internal**: users construct one of its subclasses
    (`ChanceNode`, `DecisionNode`, `UtilityNode`) rather than a bare
    `Node`. It is exported only so it can be referenced in type hints and
    ``isinstance`` checks.

    Attributes:
        name: Identifier for the node.
        parents: Names of the nodes this one relates to. Empty for a root
            node. May be edited after construction. Semantics depend on the
            node kind — see the module docstring.
    """

    name: str
    parents: tuple[str, ...] = field(default=(), kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Validate a known-field assignment, then apply it.

        Validates only the fields this base class owns (``name``, ``parents``);
        everything else passes through to ``object.__setattr__`` unchanged so
        subclass fields, internal helpers, and future additions do not break.
        Subclasses override and call ``super().__setattr__`` to add their own
        field checks.
        """
        if key == "name":
            _validate_name(value)
        elif key == "parents":
            # ``name`` is declared before ``parents`` and is therefore assigned
            # first by the generated ``__init__``; ``getattr`` covers the
            # pre-name-assignment edge case defensively.
            _validate_parents(getattr(self, "name", ""), value)
        object.__setattr__(self, key, value)

    # --- mutation helpers ---------------------------------------------------

    def add_parent(self, name: str) -> None:
        """
        Append ``name`` to parents if it is not already present.

        Going through the ``parents`` setter ensures the same validation runs
        as on construction.
        """
        if name in self.parents:
            return
        self.parents = (*self.parents, name)

    def remove_parent(self, name: str) -> None:
        """Remove ``name`` from parents; no-op if absent."""
        self.parents = tuple(p for p in self.parents if p != name)

    # --- kind / structure ---------------------------------------------------

    @property
    def kind(self) -> NodeKind:
        """
        The structural category of this node.

        Subclasses override to return their `NodeKind`. The base
        definition is abstract: a bare `Node` has no meaningful kind.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must declare its NodeKind by overriding `kind`."
        )

    @property
    def is_sink(self) -> bool:
        """
        Whether this node may not have children.

        ``False`` for chance and decision nodes. Utility nodes override to
        ``True`` — a payoff is terminal in an influence diagram.
        `InfluenceDiagram.add_edge` reads this to reject an edge that would
        give a sink a child, rather than branching on node type.
        """
        return False

    @property
    def is_discrete(self) -> bool:
        """
        Whether this node is defined over a finite, enumerable domain.

        Subclasses override. The base definition is abstract.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must declare `is_discrete` by overriding it."
        )

    # --- consistency --------------------------------------------------------

    @property
    def consistency(self) -> Consistency:
        """
        The current ``parents`` / configurable-field consistency state.

        Computed on demand from the live field values, never stored, so it
        always reflects the latest edits. Delegates to
        ``_compute_consistency``, which each subclass implements against
        its own configurable field.
        """
        return self._compute_consistency()

    def _compute_consistency(self) -> Consistency:
        """Subclass-specific consistency computation."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_compute_consistency`."
        )

    def consistency_message(self, state: Consistency) -> str:
        """
        Human-facing explanation of a non-CONSISTENT ``state``.

        Returned for `DiagramProblem` messages and `validate` errors. Keeping
        the wording on the node (rather than the diagram) means each node type
        owns the description of its own configurable field.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must declare `consistency_message`."
        )

    def validate(self) -> None:
        """
        Raise ``ValueError`` unless the node is CONSISTENT.

        This is the gate inference consumes. Editing never calls it
        automatically — an inconsistent node is a legitimate intermediate
        state while a UI or LLM is building the diagram.
        """
        state = self.consistency
        if state is Consistency.CONSISTENT:
            return
        raise ValueError(self.consistency_message(state))


# --- field validators -------------------------------------------------------


def _validate_name(name: Any) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`name` must be a non-empty string, got {name!r}.")


def _validate_parents(name: str, parents: Any) -> None:
    if not isinstance(parents, tuple):
        raise TypeError(
            f"`parents` for {name!r} must be a tuple, got {type(parents).__name__}."
        )
    if not all(isinstance(p, str) and p.strip() for p in parents):
        raise ValueError(
            f"`parents` for {name!r} must all be non-empty strings, got {parents!r}."
        )
    seen: set[str] = set()
    for parent in parents:
        if parent == name:
            raise ValueError(f"Node {name!r} cannot list itself among its parents.")
        if parent in seen:
            raise ValueError(f"`parents` for {name!r} contains duplicate {parent!r}.")
        seen.add(parent)


def _signature_matches(
    callable_: Callable[..., Any], parent_names: tuple[str, ...]
) -> bool:
    """
    Whether ``callable_`` names every entry of ``parent_names`` as a parameter.

    The guarantee the consistency gate can make is narrow: a ``CONSISTENT``
    callable *could* have used its parents. Coverage is by named parameters
    only — a parent that exists nowhere in the signature except a variadic
    ``*args`` / ``**kwargs`` does not count, because the gate cannot verify
    that a generic callable actually reads it. A ``**kwargs``-only callable is
    therefore ``CONSISTENT`` only for an empty parent set and stale for any
    wired parents; callables that resolve parents dynamically must generate a
    named-parameter signature (the gate then works for them like anyone
    else).

    A callable that names all parents but cannot be bound (a missing required
    extra parameter) is reported as not matching. A callable that *ignores*
    its parents but could have used them (an independent node) is a
    legitimate model; signature inspection cannot tell it apart from a wiring
    mistake, and the gate does not try.

    A callable that cannot be introspected (some builtins, C extensions) is
    conservatively reported as not matching.
    """
    try:
        sig = inspect.signature(callable_)
    except (TypeError, ValueError):
        return False
    named = {
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if any(parent not in named for parent in parent_names):
        return False
    try:
        sig.bind(**dict.fromkeys(parent_names))
    except TypeError:
        return False
    return True
