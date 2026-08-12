# Remaining work

Things still left to fix, collected during the review passes of the bucket-elimination, dependency, and cleanup PRs. Split into the next milestones (PR B, PR C) and independent small fixes that can land anytime.

## PR B — engine wiring (the next milestone)

1. **The milestone itself.** `engine.py` has no `solve()` verb and `infer()` has no `policy=` argument. Per the ADR and session plan: `solve(diagram) -> policy` with `_choose_solver` dispatch (bucket_elim when all-categorical + tractable, numpyro intervention-scan otherwise); `infer()` gets all-or-nothing `policy=` binding; unbound decisions raise `InferenceError`.
2. **`infer()` on a diagram with decisions/utilities crashes with a raw `AssertionError`** — `_choose_engine` routes decision/utility diagrams to the NumPyro engine (kind-based), where `to_model` hits `assert isinstance(node, ChanceNode)` (`numpyro/model.py`); an explicit `engine="ve"` on such a diagram crashes the same way in `ve/elim.py`. PR B's dispatch replaces this with a clean `InferenceError` telling the user to use `solve()` / `policy=`.
3. **`UtilityNode.is_discrete` routing story** — step 1 (honest docstring + kind-based `_choose_engine`/`_infer_numpyro` in `engine.py`) is implemented and committed. PR B completes it: the `solve()` dispatch is built on the same kind-based classification, so `is_discrete` is never read for node classification.

## PR C — separate engines with distinct return types

4. **Public per-engine entry points with honest return types.** The unified `infer()` hides result quality: a probability vector can be exact (VE) or Monte-Carlo-estimated (numpyro), and nothing in the result says which. Resolution: make each engine a public, typed entry point — the engine name becomes the quality metadata:
   - `inference.ve.query(...)` — exact probability vectors, discrete-only (already public).
   - `inference.numpyro.samples(...)` — raw posterior samples (`dict[str, jax.Array]`), any variable type (new — lift the MCMC logic out of `engine.py`).
   - `infer()` stays the ergonomic unified verb: auto-dispatch + normalization to the per-type contract (vectors for discrete, samples for continuous). It must NOT return engine-specific shapes depending on `engine=` — that would reintroduce the ambiguity.
   - Follows the same shape as PR B's `solve()` dispatch: public engines, thin unified verbs on top.

## Small fixes, unrelated to the PRs

5. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it. (Quick.)
6. **`_signature_matches` accepts `**kwargs`-only callables as "matching any parent set"** — a `dist` or `values` that silently ignores its parents is `CONSISTENT`. Documented, but it's a silent-wrongness footgun in the exact layer meant to gate inference. (Design decision, graph layer.)
7. **`infer()` result format seam** — resolved: the per-type contract (vectors for discrete, samples for continuous) is documented via `InferenceResult` and pinned by tests. The remaining "one shape" idea is answered (per-type is the only shape that preserves both exact discrete and continuous sampling); the open *exactness* dimension (can a user tell an exact answer from a sampled one?) is PR C's motivation.
