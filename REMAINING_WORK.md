# Remaining work

## Engine roadmap

Per `docs/ADR/25_08_2026_numpyro_only_engine.md`, the library is numpyro-only.

1. **NumPyro intervention-scan solver** — DONE: `decisionpy/inference/numpyro/solver.py` (Strategy B per `decision_node.md`). Enumerates the discrete policy space and estimates expected utility per policy by forward sampling with each decision resolved from its observed information set. Handles mixed/continuous diagrams; requires discrete information sets.

2. **Retire the exact engines** — DONE: `decisionpy/inference/exact/` and `inference/utils/` deleted, the `Gaussian` result type and the `exact` flags on `Marginal`/`Solution` removed, and the `engine=` plumbing gone — `infer()` and `solve()` are numpyro-only. Recoverable via git history and `docs/inference/exact_engines_plan.md` (see the ADR).

3. **Policy-as-parameters solving (Strategy A)** — deferred: continuous information sets (a decision observing a continuous parent) cannot be tabulated by the intervention scan. The parameterized-policy path (per `decision_node.md`, Strategy A) extends solving to those diagrams.

## Validation

4. **Rewrite the comparison tests against pyAgrum** — DONE: the numpyro engine is validated against pyAgrum's exact engines (LazyPropagation for categorical BNs, the ShaferShenoy LIMID solver for `solve()`, `pyagrum.clg` for continuous BNs) at Monte-Carlo tolerances, in `tests/inference/numpyro/test_compare_pyagrum*.py`. pgmpy dropped from the test dependencies — pyAgrum is the single external reference library.

5. **Docs sweep** — DONE: the living docs now describe the numpyro-only world — `decision_node.md` (Strategy B implemented, bucket elimination historical), `utility_node.md`, `backend_numpyro.md` (solver section added), `inference_strategy.md` (rewritten), `docs/README.md` (status table + code status), and the superseded ADRs (`23_07`, `13_08`) carry banner notes.

6. **pyAgrum side-by-side in the examples** — add pyAgrum's exact solutions to `solver_comparison.py` (exact LIMID vs the scan) and `categorical_bn.py` (exact VE vs enumeration), move `pyagrum` into the `dev` dependency group, and note it in `examples/README.md`.
