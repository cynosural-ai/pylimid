# Remaining work

## Engine roadmap

Per `docs/ADR/25_08_2026_numpyro_only_engine.md`, the library is numpyro-only: the exact engines (`ve`, `lg`) are retired once the NumPyro solver exists, and the CLG engine is never built.

1. **NumPyro intervention-scan solver** — `solve()` on mixed/continuous influence diagrams currently raises a clean `InferenceError`. The planned NumPyro intervention-scan / gradient-optimization path (per `decision_node.md`, Strategy B) makes those diagrams solvable. The critical path: it is the prerequisite for retiring bucket elimination and the product's differentiator.

2. **Retire the exact engines** — once the solver lands, remove in one sweep: `decisionpy/inference/exact/`, the `utils/` factor builders they alone use, the `Gaussian` result type, the `exact` flag on `Marginal`, the `engine=` plumbing in `infer()`/`solve()`, the LG fixtures and comparison tests, and the exact-result sections of the examples. The pgmpy/pyAgrum comparison tests stay. Recoverable via git history and `docs/inference/exact_engines_plan.md` (see the ADR).
