# Remaining work

Things still left to fix, collected during the review passes of the bucket-elimination, dependency, and cleanup PRs. Split into what belongs to the next milestone (PR B) and independent small fixes that can land anytime.

## PR B — engine wiring (the next milestone)

1. **The milestone itself.** `engine.py` has no `solve()` verb and `infer()` has no `policy=` argument. Per the ADR and session plan: `solve(diagram) -> policy` with `_choose_solver` dispatch (bucket_elim when all-categorical + tractable, numpyro intervention-scan otherwise); `infer()` gets all-or-nothing `policy=` binding; unbound decisions raise `InferenceError`.
2. **`infer()` on a diagram with decisions/utilities crashes with a raw `AssertionError`** — `_choose_engine` routes decision/utility diagrams to the NumPyro engine (kind-based), where `to_model` hits `assert isinstance(node, ChanceNode)` (`numpyro/model.py`); an explicit `engine="ve"` on such a diagram crashes the same way in `ve/elim.py`. PR B's dispatch replaces this with a clean `InferenceError` telling the user to use `solve()` / `policy=`.
3. **`UtilityNode.is_discrete` routing story** — step 1 (honest docstring + kind-based `_choose_engine`/`_infer_numpyro` in `engine.py`) is implemented and pending commit. PR B completes it: the `solve()` dispatch is built on the same kind-based classification, so `is_discrete` is never read for node classification.

## Small fixes, unrelated to PR B

4. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it. (Quick.)
5. **`_signature_matches` accepts `**kwargs`-only callables as "matching any parent set"** — a `dist` or `values` that silently ignores its parents is `CONSISTENT`. Documented, but it's a silent-wrongness footgun in the exact layer meant to gate inference. (Design decision, graph layer.)
6. **`infer()` result format seam** — probability vectors for discrete, raw MCMC sample arrays for continuous. One return shape would be nicer; needs a design decision (e.g. always return probability vectors, or always return samples). Could be decided alongside PR B's `infer()` signature work, but it is not required by it. (The per-type contract is documented and pinned by tests; only the "one shape" question remains open.)
