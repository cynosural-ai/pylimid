# NumPyro-only engine: retire the exact inference engines

**Date:** 2026-08-25
**Status:** Settled

> **Where this fits.** This is a *decision record* — it records a change of direction for the inference layer. It supersedes the exact-engine phases of `docs/inference/exact_engines_plan.md` (phases 2 and 4, and phase 3's framing) and amends `23_07_2026_unified_inference_architecture.md` and `13_08_2026_typed_inference_results.md` where they assume multiple inference engines.

---

## The decision

`infer()` is NumPyro-only. The exact inference engines — variable elimination (`ve`) and the linear-Gaussian engine (`lg`) — are retired, and the conditional-linear-Gaussian engine (`clg`) is not built. `solve()` keeps bucket elimination for now because it is the only working solver, and retires it once the NumPyro intervention-scan solver lands.

The motivation is the type and contract complexity the exact engines carry, not dependency weight. Each exact engine adds a result type, an extraction mechanism, and a precondition-checking path to the public API:

- the `exact` flag on `Marginal` exists only because two engines produce it;
- the CPD probe exists only because an engine other than NumPyro must recover parameters from `dist` callables;
- the `Gaussian` result type and the future `Mixture` widen the result union for a closed library whose core promise is gradient-based decision optimization on mixed diagrams.

NumPyro alone covers the model classes in scope well: parallel enumeration for discrete variables, NUTS for continuous and mixed models, and (once built) intervention-scan solving. Exactness is not load-bearing anywhere in `infer()`.

### Target contract

```
infer(diagram, query, observed=..., policy=...)
    → {name: Marginal(values) | Draws(values)}
        Marginal — discrete posterior probability vector (Monte-Carlo estimate)
        Draws    — continuous posterior draws
        no engine= parameter; nothing is ever exact

solve(diagram)
    → Solution(policy, expected_utility)
        bucket elimination (exact) until the NumPyro solver lands,
        then the NumPyro intervention-scan solver (Monte-Carlo estimate)
```

The `exact` flag disappears with the last exact engine that shares a result shape with NumPyro.

## What we are going to do (sequencing)

1. **Build the NumPyro solver** — `solve()` for mixed/continuous influence diagrams via the intervention-scan path from `docs/graph/decision_node.md` Strategy B (`numpyro.handlers.do` + `Predictive`), the REMAINING_WORK item 1. This is the critical path: it is the product's differentiator and the prerequisite for retiring bucket elimination.
2. **Simplify the library** — once the solver exists, remove the exact machinery in one sweep: `decisionpy/inference/exact/`, the `utils/` factor builders they alone use, the `Gaussian` result type, the `engine=` plumbing in `infer()`/`solve()`, the LG fixtures and comparison tests, and the exact-result sections of the examples. `Marginal` loses its `exact` flag. The validation tests against pgmpy and pyAgrum stay — external exact engines keep providing reference answers without any exact code in-repo.
3. **Re-scope REMAINING_WORK.md** and the inference ADRs to the numpyro-only world.

## Recoverability

Retiring the exact engines is deliberately reversible:

- the modules are clean, isolated trees in git history; reinstating `exact/categorical` or `exact/linear_gaussian` is a cherry-pick, not a redesign;
- `docs/inference/exact_engines_plan.md` is kept as the record of how the engines were built and validated, so a future rebuild starts from a documented design;
- the pgmpy / pyAgrum comparison tests keep exercising the same physics, so a reinstated engine can be revalidated against the same oracles.

The decision to re-add an exact engine should be made only for a concrete user need (for example, "exact answers for small models" as a product requirement), never preemptively.

## Rejected alternatives

### Two packages: numpyro core + exact extension

A package split only reduces complexity if the exact package gets its own declarative model layer (probs arrays, `(a, b, scale)` tuples) instead of `dist` callables. That eliminates the probes but forces dual model definitions — the same model expressed twice, with a silent-desync risk between them. The current code is already physically isolated (`inference/exact/` vs `inference/numpyro/`), so a future extraction is packaging surgery, not refactoring; no split is needed to keep that option open.

### Keep the exact engines as internal validation references

The comparison tests already use external exact engines (pgmpy, pyAgrum) for reference answers; in-repo exact engines add nothing to that story.

### Keep `ve` as the exact discrete default

Undone by this decision: with one engine there is no dispatch, no `exact` flag, and one honest contract ("Monte-Carlo unless it isn't sampled" is precisely the ambiguity this decision removes).
