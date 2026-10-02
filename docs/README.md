# DecisionPy docs

Design notes for the library. They capture decisions and their reasoning; module docstrings are the API reference.

The docs split into two kinds:

- **Decision records (`ADR/`)** — immutable, dated. Each records *that* a decision was made, the alternatives considered, and what changed in the tree as a result. They do not move with the code.
- **Living design notes** — the current reference for *how* a part of the library behaves and why. They are updated as the code changes.

## Decision records

1. **[`16_07_2026 - initial plan`](./ADR/16_07_2026%20-%20initial%20plan.md)** — the original vision: mixed-type influence diagrams on NumPyro, with a pluggable-backend architecture. Read this first for context on *what* we are building and why.

2. **[`17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md)** — commits to the **mutable workspace** as the library's sole container and retires the build-once alternative.

3. **[`23_07_2026_unified_inference_architecture.md`](./ADR/23_07_2026_unified_inference_architecture.md)** — commits to a unified `inference/` package with two public verbs (`infer`, `solve`). Retires the separate `backend/` directory. Its auto-dispatch story was later **superseded** by [`25_08_2026_numpyro_only_engine.md`](./ADR/25_08_2026_numpyro_only_engine.md) — the library is now NumPyro-only.

4. **[`13_08_2026_typed_inference_results.md`](./ADR/13_08_2026_typed_inference_results.md)** — committed `infer()` to typed per-entry results: `Marginal` for discrete variables, `Draws` for continuous ones. **Superseded** by [`25_08_2026_unified_posterior_result.md`](./ADR/25_08_2026_unified_posterior_result.md), which unifies both into a single `Posterior` (draws + `states`, with `marginal()` / `mean()` / `std()` / `hdi()` methods).

## Living design notes

Read in this order:

1. **[`diagram.md`](./diagram.md)** — the foundational note. The diagram is a **mutable workspace** edited by external authors (script, UI, LLM), with inference gated behind an explicit `validate()` / `snapshot()` checkpoint and the on-demand runtime `probe_discrete_parents()` check. The seven principles here govern every node and the container. Read this before the node-specific notes.

2. **[`chance_node.md`](./chance_node.md)** — settles the *distribution representation*: callable-primary (handles discrete, continuous, and mixed uniformly), with CPT tables as a diagram-scoped sugar. Applies the mutability model from `diagram.md` — `dist` is optional, consistency is derived.

3. **[`decision_node.md`](./decision_node.md)** — settles the *solving strategy*: the NumPyro intervention scan (Strategy B, implemented) for discrete decisions with discrete information sets, mixed/continuous diagrams included; bucket elimination retired (historical); policy-as-parameters (Strategy A) deferred. Includes the `DecisionNode` *representation* in `graph/` (information set, action space, consistency model).

4. **[`utility_node.md`](./utility_node.md)** — the `UtilityNode` *representation*: a callable `values(parent_assignments) -> float`, no dist/states, always a sink (enforced eagerly + defensively). Inherits the consistency model from `diagram.md`. Consumed by the intervention-scan solver's expected-utility estimation.

5. **[`multiple_utility_nodes.md`](./graph/multiple_utility_nodes.md)** — what several utility nodes mean: the additive convention, its equivalence to pyAgrum, and why true multi-objective (Pareto) optimization is out of scope.

6. **[`backend_numpyro.md`](./backend_numpyro.md)** — the NumPyro engine: turns a validated `Snapshot` into a NumPyro model. Chance nodes always; bound decisions become degenerate observed sites and utilities are skipped. `samples()` covers both forward sampling and posterior inference (enumeration / NUTS); `solvers.intervention_scan.solve` runs the intervention scan for influence diagrams. NumPyro is a declared dependency, though the graph layer itself never imports it.

7. **[`inference_strategy.md`](./inference_strategy.md)** — analysis of inference backends for the full influence diagram roadmap. Decision: NumPyro as the only engine (see [`25_08_2026_numpyro_only_engine.md`](./ADR/25_08_2026_numpyro_only_engine.md)), with pyAgrum's exact solvers as the external validation reference.

8. **[`bucket_elim.md`](./inference/bucket_elim.md)** — **historical reference.** The exact influence-diagram solver that served as the v0 solve() path: algorithm (additive utilities, reverse-topological elimination, decision max-out), the policy representation, and the supported scope. Retired with the exact engines; its history is in git.

9. **[`solver_algorithms.md`](./inference/solver_algorithms.md)** — the solver family and the papers behind it: the intervention scan (implemented), the batched scan (implemented), and backward induction / single policy updating (planned), with the LIMID theory (Lauritzen–Nilsson), the regularity/memory-arcs discussion, and the Monte-Carlo decision-analysis references.

10. **[`backward_induction.md`](./inference/backward_induction.md)** — the Level 2 estimator design (planned): the per-decision `Q(a, k)` conditional expectation, stratified grouping over information-set assignments, the regularity gate, variance expectations, and the validation plan against pyAgrum.

## Status of each decision

| Topic                | Status      | Where                              |
| -------------------- | ----------- | ---------------------------------- |
| Overall vision       | Settled     | `ADR/16_07_2026 - initial plan`    |
| Mutable workspace + validation gate | Settled | `diagram.md` |
| Build-once vs. mutable (commit) | Settled — mutable | `ADR/17_07_2026_mutability_design_decision.md` |
| Chance-node distribution form | Settled | `chance_node.md`             |
| Decision-node representation | Settled — graph layer | `decision_node.md` |
| Utility-node representation | Settled — graph layer | `utility_node.md` |
| Multiple utility nodes | Settled — additive scalarization, same convention as pyAgrum; true multi-objective (Pareto) out of scope | `multiple_utility_nodes.md` |
| Decision-node solving strategy | Settled — NumPyro intervention scan, batched scan, and backward induction (regular diagrams) implemented; SPU for non-regular LIMIDs deferred; bucket elimination retired (historical); Strategy A deferred | `decision_node.md`, `solver_algorithms.md`, `ADR/25_08_2026_numpyro_only_engine.md` |
| NumPyro translator   | Chance nodes; bound decisions as observed sites, utilities skipped — `samples()` forward + posterior (enumeration / NUTS), `solvers.intervention_scan.solve` intervention scan | `backend_numpyro.md` |
| Inference strategy   | Settled — NumPyro-only | `ADR/25_08_2026_numpyro_only_engine.md` |
| Unified inference API | Settled — `infer()` / `solve()`, NumPyro-only | `ADR/23_07_2026_unified_inference_architecture.md`, `ADR/25_08_2026_numpyro_only_engine.md` |
| `infer()` result format | Settled — one `Posterior` per query (draws + `states`; `marginal()`/`mean()`/`std()`/`hdi()`) | `ADR/25_08_2026_unified_posterior_result.md` |
| `from_cpt` sugar     | Deferred    | `chance_node.md` (resolved q)      |
| Parametric learning (`fit`) | Deferred | `diagram.md`                 |

## Code status

`decisionpy/` implements the mutable workspace and a first inference engine:

- `graph/node.py` — the shared `Node` base: `name`/`parents`, field validation, and the `UNCONFIGURED`/`STALE`/`CONSISTENT` consistency gate that every node type inherits. Also defines `NodeKind` and `Consistency`.
- `graph/chance_node.py`, `graph/decision_node.py`, `graph/utility_node.py` — the three node types, all subclassing `Node`. A diagram with only chance nodes is a Bayesian network; adding a decision or utility node makes it an influence diagram (see `diagram.md`, `decision_node.md`, `utility_node.md`).
- `graph/diagram.py` — the mutable container and `Snapshot` (see `diagram.md`); `graph/validation.py` — the `DiagramProblem` / `ProblemKind` model shared by `validate()` and `probe_discrete_parents()`. Exports `Node`, `NodeKind`, `Consistency`, the three node types, and the problem model from `decisionpy.graph`.
- `inference/numpyro/` — the NumPyro engine: translates a `Snapshot` into a NumPyro model, with public `samples()` — prior draws with no observations, posterior draws (enumeration / NUTS) with observations, as raw JAX arrays — and `solvers.intervention_scan.solve` — the intervention-scan influence-diagram solver. Bound decisions clamp to their policy values; utilities are skipped. NumPyro is a declared dependency, though nothing in `decisionpy.graph` imports it.
- `inference/result.py` — the typed results: `Posterior` (raw draws plus `states`, with kind-guarded `marginal()` / `mean()` / `std()` / `hdi()` methods), `Solution`, `Policy`, `InferenceResult`.
- `inference/engine.py` — unified `infer()` / `solve()` entry-points (NumPyro-only). `infer()` returns one `Posterior` per query variable: raw draws plus the variable's `states` (`None` for continuous); `marginal()` bincounts a discrete posterior into its probability vector and raises on a continuous one. `infer()` takes an all-or-nothing `policy=` binding (unbound decisions raise a clean `InferenceError`); `solve(diagram, method="auto" | "backward_induction" | "scan")` is the solver front door for mixed/continuous influence diagrams with discrete information sets: `"auto"` runs backward induction when the diagram is solvable and the batched scan otherwise. The main names are re-exported from the top-level `decisionpy` package.

The exact engines (variable elimination, bucket elimination, the linear-Gaussian engine) were built and retired in favor of the NumPyro-only engine; their history lives in git (see [`25_08_2026_numpyro_only_engine.md`](./ADR/25_08_2026_numpyro_only_engine.md)).

The build-once prototype that preceded this design has been removed; see [`ADR/17_07_2026_mutability_design_decision.md`](./ADR/17_07_2026_mutability_design_decision.md).
