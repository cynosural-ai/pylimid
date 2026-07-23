# DecisionPy docs

Design notes for the library. They capture decisions and their reasoning;
module docstrings are the API reference.

The docs split into two kinds:

- **Decision records (`ADR/`)** — immutable, dated. Each records *that* a
  decision was made, the alternatives considered, and what changed in the tree
  as a result. They do not move with the code.
- **Living design notes** — the current reference for *how* a part of the
  library behaves and why. They are updated as the code changes.

## Decision records

1. **[`16_07_2026 - initial plan`](./ADR/16_07_2026%20-%20initial%20plan.md)** —
   the original vision: mixed-type influence diagrams on NumPyro, with a
   pluggable-backend architecture. Read this first for context on *what* we
   are building and why.

2. **[`17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md)**
   — commits to the **mutable workspace** as the library's sole container and
   retires the build-once alternative.

## Living design notes

Read in this order:

1. **[`diagram.md`](./diagram.md)** — the foundational note. The diagram is a
   **mutable workspace** edited by external authors (script, UI, LLM), with
   inference gated behind an explicit `validate()` / `snapshot()` checkpoint.
   The seven principles here govern every node and the container. Read this
   before the node-specific notes.

2. **[`chance_node.md`](./chance_node.md)** — settles the *distribution
   representation*: callable-primary (handles discrete, continuous, and mixed
   uniformly), with CPT tables as a diagram-scoped sugar. Applies the
   mutability model from `diagram.md` — `dist` is optional, consistency is
   derived.

3. **[`decision_node.md`](./decision_node.md)** — settles the *solving
   strategy*: intervention-scan (Strategy B) for v0 discrete decisions,
   policy-as-parameters (Strategy A) for continuous, composed by a per-node
   dispatch abstraction for mixed. Inherits the consistency model from
   `diagram.md`.

## Status of each decision

| Topic                | Status      | Where                              |
| -------------------- | ----------- | ---------------------------------- |
| Overall vision       | Settled     | `ADR/16_07_2026 - initial plan`    |
| Mutable workspace + validation gate | Settled | `diagram.md` |
| Build-once vs. mutable (commit) | Settled — mutable | `ADR/17_07_2026_mutability_design_decision.md` |
| Chance-node distribution form | Settled | `chance_node.md`             |
| Decision-node solving strategy | Settled (v0 = B) | `decision_node.md`    |
| NumPyro translator   | Not started | —                                  |
| `from_cpt` sugar     | Deferred    | `chance_node.md` (resolved q)      |
| Parametric learning (`fit`) | Deferred | `diagram.md`                 |

## Code status

`src/decisionpy/graph/` implements the mutable workspace:

- `chance_node.py` + `diagram.py` — the mutable node and container (see
  `diagram.md`). Both are exported from `decisionpy.graph`.

The build-once prototype that preceded this design has been removed; see
[`ADR/17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md).
