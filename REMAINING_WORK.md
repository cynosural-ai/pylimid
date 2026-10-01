# Remaining work

Per `docs/ADR/25_08_2026_numpyro_only_engine.md`, the library is numpyro-only.

## Solver roadmap

1. **SPU iteration (non-soluble LIMIDs)** — re-apply the per-decision update until no rule changes (a local optimum); once it lands the `is_solvable` gate stops being load-bearing. The estimator design (sampling, grouping, variance, validation) is in `docs/inference/backward_induction.md`.

2. **Policy-as-parameters solving (Strategy A)** — deferred: continuous information sets (a decision observing a continuous parent) cannot be tabulated by the intervention scan. The parameterized-policy path (per `decision_node.md`, Strategy A) extends solving to those diagrams. The feasibility survey, methods, and literature map for that path are in `docs/continuous_decisions/`.
