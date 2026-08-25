# Mutable influence diagram — design principles

Design note for the container in `diagram.py` and the node base in `node.py`.
Captures *why* the container behaves the way it does, not the API reference
(which lives in the module docstrings).

> **Where this fits.** This is the foundational design note — the mutability
> and validation model here governs every node type. See
> [`README.md`](./README.md) for the doc index and reading order. The
> node-specific notes ([`chance_node.md`](./chance_node.md),
> [`decision_node.md`](./decision_node.md), [`utility_node.md`](./utility_node.md))
> build on the principles below.

---

## The one requirement that decides the shape

The diagram is edited by **external authors** — a script, a GUI, or an LLM
driving the model over a tool interface. None of them can produce a complete,
valid influence diagram in a single atomic step. Construction is necessarily
incremental and, during construction, the diagram is necessarily *incomplete*:
a node may exist before its parents are wired; a parent may be referenced
before it is added; a distribution may be unset while structure is still being
decided.

A data structure that cannot represent its own intermediate states forces the
author to maintain a parallel shadow state and synthesize a valid object on
every action. That is a second system that will drift from the real one. So
the diagram must be **mutable**, and it must tolerate incompleteness as a
first-class state.

---

## Principle 1 — editing and inference are separated by an explicit gate

The diagram is allowed to be unsound at all times. Inference consumes a
**validated snapshot**, never the live workspace. `validate()` is the gate.

This mirrors the working-tree / commit split in Git, or live tables / snapshot
isolation in a database: the source of truth is always editable; the consumer
sees an immutable, checked view.

Concretely: `add_node`, `add_edge`, `set_dist` never reject an edit for being
incomplete (only for being *malformed* — see Principle 2). `validate()` /
`snapshot()` are the only operations that demand soundness.

---

## Principle 2 — two layers of checking, with different jobs

| Layer              | When it runs        | What it catches                              |
| ------------------ | ------------------- | -------------------------------------------- |
| Field validation   | every assignment    | a *malformed* value — empty name, non-callable dist, duplicate parent |
| Cross-field / graph | `validate()` only  | an *incomplete or inconsistent* state — dangling parent, stale dist signature, cycle |
| Runtime probe      | `probe_discrete_parents()` on demand | a callable that ignores a discrete parent (same output for every parent value) |

A malformed value (empty name) is never a useful intermediate state, so it is
rejected the instant it is set. An incomplete state (dist not yet configured)
*is* a useful intermediate, so it is permitted during editing and gated only at
inference time. Conflating the two — rejecting incompleteness at set-time —
breaks incremental construction and is the mistake the build-once design made.

`validate()` is structural: it inspects signatures and graph shape but never
executes a `dist`. `probe_discrete_parents()` is the runtime counterpart — it
executes each callable (explicitly, on demand, never automatically) and flags
a discrete parent whose value never changes the output, the classic silent
mistake of a probability table with too few rows under JAX's clamping
semantics. The two share the `DiagramProblem` reporting model in
`decisionpy.graph.validation`. A probe finding is a warning, not an error: a
deliberately independent node looks identical.

---

## Principle 3 — validation reports, it does not raise on the first problem

`validate()` returns a *list* of `DiagramProblem` records, not a single
exception. An author editing through a UI or over a tool interface wants to see
every issue at once ("WetGrass has a dangling parent *and* its dist is stale"),
not fix them one raise at a time. Structured records (`kind`, `node`,
`message`) let the caller render or act without parsing prose.

---

## Principle 4 — structural invariants are enforced eagerly on the happy path, defensively elsewhere

Two structural invariants are "never a useful intermediate state" and get this
two-layer treatment:

- **Acyclicity.** A cycle is never useful. `add_edge` rejects an edge that
  would close a cycle before applying it. But a node is independently mutable
  through its own `add_parent`, which bypasses the diagram, so `validate()`
  runs a defensive cycle check as the backstop.
- **Utility nodes are sinks.** A payoff with a child is never valid. `add_edge`
  rejects an edge whose parent is a utility node (read via `node.is_sink`).
  The same direct-mutation bypass applies, so `validate()` reports any utility
  node that has nonetheless acquired a child as `ProblemKind.UTILITY_NOT_SINK`.

Two layers because there are two mutation paths; neither alone is sufficient.

---

## Principle 5 — the node carries its own consistency; the diagram aggregates it

Per-node state (`UNCONFIGURED` / `STALE` / `CONSISTENT`) is computed by the
node, from its own fields, on demand — never stored, never cached. The diagram
does not duplicate that logic; it asks each node for its consistency and rolls
the answers up into `DiagramProblem`s. One place owns each piece of knowledge.

---

## Principle 6 — adjacency is derived, not stored

`children_of` scans the node set on each call rather than maintaining a
parallel parent→child index. The trade is simplicity and correctness (no index
to desync when a node is mutated through its own API) against O(n) per lookup.
Acceptable at prototype scale; a maintained index is a localized change if it
later matters.

---

## Principle 7 — the snapshot is logical, not deep-frozen

`snapshot()` returns references to the still-mutable nodes in topological
order. Editing the diagram after snapshotting invalidates the snapshot.
Freezing the nodes is a documented future tightening; the v0 contract is that
the caller does not mutate captured nodes before inference runs.

---

## What this deliberately is not

- **Not a build-once container.** Nodes may be added before their parents,
  edges rewired, distributions filled in at any time. The build-once prototype
  that preceded this design has been removed — see
  [`17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md)
  for the decision.
- **Not mutable at random.** Field-level validation rejects malformed values
  immediately; only completeness is deferred.
- **Not a parametric learner.** Parameter fitting (MLE, EM, SGD over CPTs) is
  a future concern and will likely take the form of a functional
  `fit(diagram, data) -> diagram` rather than in-place mutation of closed-over
  params. The current design does not commit to either, but does not block
  either.
