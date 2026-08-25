# Remaining work

## Engine roadmap

Per `docs/ADR/25_08_2026_numpyro_only_engine.md`, the library is numpyro-only: the exact engines (`ve`, `lg`) are retired once the NumPyro solver exists, and the CLG engine is never built.

1. **NumPyro intervention-scan solver** — DONE: `decisionpy/inference/numpyro/solver.py` (Strategy B per `decision_node.md`). Enumerates the discrete policy space and estimates expected utility per policy by forward sampling with each decision resolved from its observed information set. Handles mixed/continuous diagrams; requires discrete information sets (a continuous information set is the policy-as-parameters path, deferred). Opt-in via `solve(engine="numpyro")`; bucket elimination remains the default until item 2.

2. **Retire the exact engines** — remove in one sweep: `decisionpy/inference/exact/`, the `utils/` factor builders they alone use, the `Gaussian` result type, the `exact` flag on `Marginal`, the `engine=` plumbing in `infer()`/`solve()`, the LG fixtures and comparison tests, and the exact-result sections of the examples. After this, `solve()` defaults to the NumPyro solver and the pgmpy/pyAgrum comparison tests stay. Recoverable via git history and `docs/inference/exact_engines_plan.md` (see `docs/ADR/25_08_2026_numpyro_only_engine.md`).
