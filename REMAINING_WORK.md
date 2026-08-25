# Remaining work

## Engine roadmap

Per `docs/ADR/25_08_2026_numpyro_only_engine.md`, the library is numpyro-only.

1. **NumPyro intervention-scan solver** — DONE: `decisionpy/inference/numpyro/solver.py` (Strategy B per `decision_node.md`). Enumerates the discrete policy space and estimates expected utility per policy by forward sampling with each decision resolved from its observed information set. Handles mixed/continuous diagrams; requires discrete information sets.

2. **Retire the exact engines** — DONE: `decisionpy/inference/exact/` and `inference/utils/` deleted, the `Gaussian` result type and the `exact` flags on `Marginal`/`Solution` removed, and the `engine=` plumbing gone — `infer()` and `solve()` are numpyro-only. Recoverable via git history and `docs/inference/exact_engines_plan.md` (see the ADR).

3. **Policy-as-parameters solving (Strategy A)** — deferred: continuous information sets (a decision observing a continuous parent) cannot be tabulated by the intervention scan. The parameterized-policy path (per `decision_node.md`, Strategy A) extends solving to those diagrams.

## Solver performance

4. **Batch the policy scan (Level 1)** — the intervention scan evaluates each policy with its own `jax.vmap` call, so JAX re-traces per policy (measured ~260ms per 500-sample evaluation, ~60% of it traced `arr[idx]` dynamic-slice machinery from user dist callables). Evaluate all policies in one vmapped call with the policy arrays as a leading batch dimension and jit the forward step once: one trace, one dispatch for the whole solve — expected 10-50× on the scan fixtures, same semantics.

5. **Per-decision backward induction (Level 2)** — replace the full policy-product enumeration with choosing one decision at a time in reverse order, estimating E[U | info] per action by stratified forward sampling. Removes the exponential in the policy space; introduces estimator variance per info-set group, so it needs careful design after Level 1.

## Validation

6. **Rewrite the comparison tests against pyAgrum** — DONE: the numpyro engine is validated against pyAgrum's exact engines (LazyPropagation for categorical BNs, the ShaferShenoy LIMID solver for `solve()`, `pyagrum.clg` for continuous BNs) at Monte-Carlo tolerances, in `tests/inference/numpyro/test_compare_pyagrum*.py`. pgmpy dropped from the test dependencies — pyAgrum is the single external reference library.

7. **Docs sweep** — DONE: the living docs now describe the numpyro-only world — `decision_node.md` (Strategy B implemented, bucket elimination historical), `utility_node.md`, `backend_numpyro.md` (solver section added), `inference_strategy.md` (rewritten), `docs/README.md` (status table + code status), and the superseded ADRs (`23_07`, `13_08`) carry banner notes.

8. **pyAgrum side-by-side in the examples** — DONE: `solver_comparison.py` shows pyAgrum's exact LIMID solution (MEU 78.0, same policy) next to the scan's estimate, and `categorical_bn.py` shows pyAgrum's exact posterior next to the enumeration estimate. pyagrum moved into the `dev` dependency group and noted in `examples/README.md`.
