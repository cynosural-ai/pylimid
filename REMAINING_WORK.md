# Remaining work

Things still left to fix, collected during the review passes of the bucket-elimination, dependency, and cleanup PRs. Grouped roughly by effort.

## Quick fixes (stale references, one-liners)

1. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it.

## Code seams (design-level, worth deciding before fixing)

2. **PR B — engine wiring (the next milestone).** `engine.py` has no `solve()` verb and `infer()` has no `policy=` argument. Per the ADR and session plan: `solve(diagram) -> policy` with `_choose_solver` dispatch (bucket_elim when all-categorical + tractable, numpyro intervention-scan otherwise); `infer()` gets all-or-nothing `policy=` binding; unbound decisions raise `InferenceError`.
3. **`infer()` on a diagram with decisions/utilities crashes with a raw `AssertionError`** — `_choose_engine` routes purely on `is_discrete`, so an all-categorical ID reaches VE, which hits `assert isinstance(node, ChanceNode)` (`ve/elim.py`), and a utility-containing diagram reaches the NumPyro translator with the same assert. Should be a clean `InferenceError` telling the user to use `solve()` / `policy=`. PR B fixes this.
4. **`_signature_matches` accepts `**kwargs`-only callables as "matching any parent set"** — a `dist` or `values` that silently ignores its parents is `CONSISTENT`. Documented, but it's a silent-wrongness footgun in the exact layer meant to gate inference.
5. **`UtilityNode.is_discrete` returns `False` as a routing proxy** — semantically "not a BN-engine input", not "not finite". Works today, but conflates two meanings; the first engine that reads `is_discrete` for its own purposes will trip on it.
6. **`infer()` result format seam** — probability vectors for discrete, raw MCMC sample arrays for continuous. One return shape would be nicer; needs a design decision (e.g. always return probability vectors, or always return samples).
7. **`docs/graph/decision_node.md` recommends Strategy B (intervention-scan) for v0** — the actual v0 solver is exact bucket elimination. The "Recommendation: B for v0" section still reads as the committed path; reconcile it with what was implemented (the solving-status wording is updated, the recommendation narrative is not).
