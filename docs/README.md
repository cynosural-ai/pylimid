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

3. **[`23_07_2026_unified_inference_architecture.md`](./ADR/23_07_2026_unified_inference_architecture.md)**
   — commits to a unified `inference/` package with two public verbs (`infer`,
   `solve`) and auto-dispatch across graph-native and PPL engines. Retires the
   separate `backend/` directory.

4. **[`13_08_2026_typed_inference_results.md`](./ADR/13_08_2026_typed_inference_results.md)**
   — committed `infer()` to typed per-entry results: `Marginal` for discrete
   variables, `Draws` for continuous ones. **Superseded** by
   [`25_08_2026_unified_posterior_result.md`](./ADR/25_08_2026_unified_posterior_result.md),
   which unifies both into a single `Posterior` (draws + `states`, with
   `marginal()` / `mean()` / `std()` / `hdi()` methods).

## Living design notes

Read in this order:

1. **[`diagram.md`](./diagram.md)** — the foundational note. The diagram is a
   **mutable workspace** edited by external authors (script, UI, LLM), with
   inference gated behind an explicit `validate()` / `snapshot()` checkpoint
   and the on-demand runtime `probe_discrete_parents()` check. The seven
   principles here govern every node and the container. Read this before the
   node-specific notes.

2. **[`chance_node.md`](./chance_node.md)** — settles the *distribution
   representation*: callable-primary (handles discrete, continuous, and mixed
   uniformly), with CPT tables as a diagram-scoped sugar. Applies the
   mutability model from `diagram.md` — `dist` is optional, consistency is
   derived.

3. **[`decision_node.md`](./decision_node.md)** — settles the *solving
   strategy*: bucket elimination for exact discrete diagrams (implemented),
   intervention-scan (Strategy B) as the planned NumPyro path for
   mixed/continuous, policy-as-parameters (Strategy A) deferred. Includes the
   `DecisionNode` *representation* in `graph/` (information set, action space,
   consistency model).

4. **[`utility_node.md`](./utility_node.md)** — the `UtilityNode` *representation*:
   a callable `values(parent_assignments) -> float`, no dist/states, always a
   sink (enforced eagerly + defensively). Inherits the consistency model from
   `diagram.md`. Consumed by the exact solver's expected-utility computation.

5. **[`backend_numpyro.md`](./backend_numpyro.md)** — the NumPyro translator:
   turns a validated `Snapshot` into a NumPyro model. Chance nodes always;
   bound decisions become degenerate observed sites and utilities are skipped.
   `samples()` covers both forward sampling and posterior inference (exact
   enumeration / NUTS). NumPyro is a declared dependency, though the graph
   layer itself never imports it.

6. **[`inference_strategy.md`](./inference_strategy.md)** — analysis of
   inference backends for the full influence diagram roadmap: variable
   elimination and NumPyro (NUTS/SVI). Decision: VE for exact discrete,
   NumPyro as the primary engine for mixed-type + gradient-based decision
   optimization.

7. **[`bucket_elim.md`](./inference/bucket_elim.md)** — the exact
   influence-diagram solver: algorithm (additive utilities, reverse-topological
   elimination, decision max-out), the policy representation, and the supported
   scope (all-categorical, regular structures).

## Status of each decision

| Topic                | Status      | Where                              |
| -------------------- | ----------- | ---------------------------------- |
| Overall vision       | Settled     | `ADR/16_07_2026 - initial plan`    |
| Mutable workspace + validation gate | Settled | `diagram.md` |
| Build-once vs. mutable (commit) | Settled — mutable | `ADR/17_07_2026_mutability_design_decision.md` |
| Chance-node distribution form | Settled | `chance_node.md`             |
| Decision-node representation | Settled — graph layer | `decision_node.md` |
| Utility-node representation | Settled — graph layer | `utility_node.md` |
| Decision-node solving strategy | Settled — exact bucket elimination implemented; Strategy B (intervention-scan) planned for the NumPyro path | `decision_node.md`, `bucket_elim.md` |
| NumPyro translator   | Chance nodes; bound decisions as observed sites, utilities skipped — `samples()` forward + posterior (enumeration / NUTS) | `backend_numpyro.md` |
| Inference strategy   | Settled — VE for exact discrete, NumPyro for mixed-type | `inference_strategy.md` |
| Unified inference API | Settled — `infer()` / `solve()` with auto-dispatch | `ADR/23_07_2026_unified_inference_architecture.md` |
| `infer()` result format | Settled — one `Posterior` per query (draws + `states`; `marginal()`/`mean()`/`std()`/`hdi()`) | `ADR/25_08_2026_unified_posterior_result.md` |
| `from_cpt` sugar     | Deferred    | `chance_node.md` (resolved q)      |
| Parametric learning (`fit`) | Deferred | `diagram.md`                 |

## Code status

`decisionpy/` implements the mutable workspace and a first inference engine:

- `graph/node.py` — the shared `Node` base: `name`/`parents`, field validation,
  and the `UNCONFIGURED`/`STALE`/`CONSISTENT` consistency gate that every node
  type inherits. Also defines `NodeKind` and `Consistency`.
- `graph/chance_node.py`, `graph/decision_node.py`, `graph/utility_node.py` —
  the three node types, all subclassing `Node`. A diagram with only chance
  nodes is a Bayesian network; adding a decision or utility node makes it an
  influence diagram (see `diagram.md`, `decision_node.md`, `utility_node.md`).
- `graph/diagram.py` — the mutable container and `Snapshot` (see `diagram.md`);
  `graph/validation.py` — the `DiagramProblem` / `ProblemKind` model shared by
  `validate()` and `probe_discrete_parents()`. Exports `Node`, `NodeKind`,
  `Consistency`, the three node types, and the problem model from
  `decisionpy.graph`.
- `inference/ve/` — exact discrete inference via variable elimination:
  public `query()` returning exact probability vectors
  (see `inference_strategy.md`).
- `inference/id/` — exact influence-diagram solving via bucket elimination:
  optimal discrete policies (`solve()` core) for all-categorical LIMIDs,
  numpy-only (see `bucket_elim.md`).
- `inference/numpyro/` — NumPyro bridge: translates a `Snapshot` into a NumPyro
  model, with public `samples()` — prior draws with no observations, posterior
  draws (exact enumeration / NUTS) with observations, as raw JAX arrays.
  Bound decisions clamp to their policy values; utilities are skipped.
  NumPyro is a declared dependency, though nothing in `decisionpy.graph`
  imports it.
- `inference/engine.py` — unified `infer()` / `solve()` entry-points
  (NumPyro-only). `infer()` returns one `Posterior` per query variable: raw
  draws plus the variable's `states` (`None` for continuous); `marginal()`
  bincounts a discrete posterior into its probability vector and raises on a
  continuous one. `infer()` takes an all-or-nothing `policy=` binding
  (unbound decisions raise a clean `InferenceError`); `solve()` runs the
  NumPyro intervention-scan solver for mixed/continuous influence diagrams
  with discrete information sets.

The build-once prototype that preceded this design has been removed; see
[`ADR/17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md).
