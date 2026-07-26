"""
Chance node — a random variable conditioned on its parents.

Implements the mutable node design settled in ``docs/chance_node.md`` and
``docs/diagram_mutable_design.md``; see
``docs/17_07_2026_mutability_design_decision.md`` for the decision that makes
this the library's sole node container.

Mutable by design
-----------------
The node is a workspace value that may be edited incrementally — add a parent,
set the distribution, swap states — by an external author: a script, a UI, or
an LLM driving the diagram over a tool interface. Because of that, a node is
allowed to exist in an *inconsistent* state: its ``dist`` may be unset, or its
signature may not yet match its parents after an edge change.

Two layers of checking reflect this:

- **Field-level validation** runs on every assignment (including during
  construction). A bad value — empty name, duplicate parent, non-callable dist
  — is rejected the moment it is set.
- **Cross-field consistency** (``dist`` signature vs ``parents``) is *not*
  enforced on assignment. It is queryable via :attr:`ChanceNode.consistency`
  and gated explicitly via :meth:`ChanceNode.validate`, which is the checkpoint
  inference runs against. Editing pauses wherever it likes; inference requires
  a ``CONSISTENT`` node.

The ``dist`` callable receives resolved parent values as **keyword arguments**
(keyed by parent name), so it is tied to parent *names*, which are stable,
rather than parent *order*, which is incidental.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Factory returning a distribution object given resolved parent values.
#:
#: The graph layer is opaque to the concrete distribution type a backend uses,
#: so the return is typed ``Any`` here. The backend's translator narrows it.
DistFactory = Callable[..., Any]


class Consistency(Enum):
    """
    State of the ``parents`` / ``dist`` relationship on a chance node.

    :var UNCONFIGURED: ``dist`` is ``None`` — structure exists, the
        distribution has not yet been supplied.
    :var STALE: ``dist`` is set but its signature does not match ``parents`` —
        typically because a parent was added or removed since ``dist`` was
        last configured.
    :var CONSISTENT: ``dist`` is set and accepts ``parents`` as keyword
        arguments — the node is ready to be translated / inferred.
    """

    UNCONFIGURED = "unconfigured"
    STALE = "stale"
    CONSISTENT = "consistent"


@dataclass
class ChanceNode:
    """
    A random variable in an influence diagram: ``P(name | parents)``.

    The node is mutable. Field-level validation runs on every assignment
    (including during construction), so a bad value is rejected at the moment
    it is set rather than later. Cross-field consistency (``dist`` signature
    vs ``parents``) is *not* enforced on assignment — it is queryable via
    :attr:`consistency` and gated via :meth:`validate`.

    :param str name: Identifier for the node.
    :param tuple[str, ...] parents: Names of the nodes this one conditionally
        depends on. Empty for a root node. May be edited after construction.
    :param DistFactory | None dist: Callable invoked as
        ``dist(**parent_values)`` returning the node's distribution. ``None``
        (the default) marks the node as not-yet-configured.
    :param tuple[str, ...] | None states: Human-facing labels for a discrete
        node's outcomes. ``None`` marks the node as continuous. Values flowing
        through the graph are integer indices into this tuple.
    """

    name: str
    parents: tuple[str, ...] = field(default=(), kw_only=True)
    dist: DistFactory | None = field(default=None, kw_only=True)
    states: tuple[str, ...] | None = field(default=None, kw_only=True)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Validate a known-field assignment, then apply it.

        Unknown attributes pass through unchanged so internal helpers and
        future fields do not break.
        """
        if key == "name":
            _validate_name(value)
        elif key == "parents":
            # ``name`` is declared before ``parents`` and is therefore assigned
            # first by the generated ``__init__``; ``getattr`` covers the
            # pre-name-assignment edge case defensively.
            _validate_parents(getattr(self, "name", ""), value)
        elif key == "dist":
            _validate_dist(value)
        elif key == "states":
            _validate_states(value)
        object.__setattr__(self, key, value)

    # --- mutation helpers ---------------------------------------------------

    def add_parent(self, name: str) -> None:
        """
        Append ``name`` to :attr:`parents` if it is not already present.

        Going through the ``parents`` setter ensures the same validation runs
        as on construction.
        """
        if name in self.parents:
            return
        self.parents = (*self.parents, name)

    def remove_parent(self, name: str) -> None:
        """Remove ``name`` from :attr:`parents`; no-op if absent."""
        self.parents = tuple(p for p in self.parents if p != name)

    # --- consistency --------------------------------------------------------

    @property
    def consistency(self) -> Consistency:
        """
        The current ``parents`` / ``dist`` consistency state.

        Computed on demand from the live field values, never stored, so it
        always reflects the latest edits.
        """
        if self.dist is None:
            return Consistency.UNCONFIGURED
        if _signature_matches(self.dist, self.parents):
            return Consistency.CONSISTENT
        return Consistency.STALE

    def validate(self) -> None:
        """
        Raise ``ValueError`` unless the node is :attr:`Consistency.CONSISTENT`.

        This is the gate inference consumes. Editing never calls it
        automatically — an inconsistent node is a legitimate intermediate
        state while a UI or LLM is building the diagram.
        """
        state = self.consistency
        if state is Consistency.UNCONFIGURED:
            raise ValueError(f"Node {self.name!r} has no `dist` configured yet.")
        if state is Consistency.STALE:
            raise ValueError(
                f"Node {self.name!r}'s `dist` signature does not match its "
                f"parents {self.parents!r}; reconfigure `dist` after changing "
                f"the parent set."
            )

    @property
    def is_discrete(self) -> bool:
        """Whether the node has a declared set of discrete states."""
        return self.states is not None


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


def _signature_matches(dist: DistFactory, parent_names: tuple[str, ...]) -> bool:
    """
    Whether ``dist`` accepts ``parent_names`` as keyword arguments.

    A ``**kwargs``-only callable matches any parent set (bind accepts
    arbitrary kwargs). A callable whose parameters do not line up with
    ``parent_names`` — missing a parent, or carrying an extra parameter —
    fails ``bind`` and is reported as stale.

    A callable that cannot be introspected (some builtins, C extensions) is
    conservatively reported as not matching. Callers that want to bypass
    signature checking should accept ``**kwargs``.
    """
    try:
        sig = inspect.signature(dist)
    except (TypeError, ValueError):
        return False
    try:
        sig.bind(**dict.fromkeys(parent_names))
    except TypeError:
        return False
    return True
