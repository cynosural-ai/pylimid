# Remaining work

Things still left to fix, collected during the review passes of the bucket-elimination, dependency, and cleanup PRs. Grouped roughly by effort.

## Quick fixes (stale references, one-liners)

1. **`decisionpy/inference/__init__.py` docstring** — points at `decisionpy.inference._api`, a module that does not exist (dispatch lives in `engine.py`). Fix the reference.
2. **`tests/inference/test_variable_elim.py` module docstring** — says `:mod:`decisionpy.inference.variable_elim``, also a nonexistent module (that was the ADR's planned name; the real module is `decisionpy.inference.ve`).
3. **`docs/README.md` status table** — "Decision-node solving strategy | Settled (v0 = B); implementation deferred" is stale: bucket elimination landed in PR #6. Same for the "NumPyro translator" row, which now also does conditional inference / MCMC / SVI beyond forward sampling.
4. **`docs/graph/decision_node.md` and `docs/graph/utility_node.md`** — still say solving is "the next milestone / not in this step". Bucket elimination exists; the docs should point at the `inference/id` solver instead of describing it as deferred.
5. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it.
6. **`inference/ve/elim.py` comment inaccuracy** — the comment in `query()` says `cardinalities` "guarantees every node has `states`, which only chance nodes can be". Wrong: decision nodes have `states` too. The `assert isinstance(node, ChanceNode)` is what actually narrows; reword the comment.
7. **`docs/sessions/SESSION_2026_08_09.md`** — the living session doc has not been updated with the later work (mandatory dependencies, cleanup pass, vectorized NumPyro inference, pyAgrum validation details are there but the rest is not). Bring it up to date.
8. **No living design note for bucket elimination** — `docs/README.md` lists a living note per engine, but the `inference/id` solver only exists in the session doc. A short `docs/inference/bucket_elim.md` (algorithm, policy representation, supported scope / non-regular guard) would complete the docs set.

## Code seams (design-level, worth deciding before fixing)

9. **PR B — engine wiring (the next milestone).** `engine.py` has no `solve()` verb and `infer()` has no `policy=` argument. Per the ADR and session plan: `solve(diagram) -> policy` with `_choose_solver` dispatch (bucket_elim when all-categorical + tractable, numpyro intervention-scan otherwise); `infer()` gets all-or-nothing `policy=` binding; unbound decisions raise `InferenceError`.
10. **`infer()` on a diagram with decisions/utilities crashes with a raw `AssertionError`** — `_choose_engine` routes purely on `is_discrete`, so an all-categorical ID reaches VE, which hits `assert isinstance(node, ChanceNode)` (`ve/elim.py`), and a utility-containing diagram reaches the NumPyro translator with the same assert. Should be a clean `InferenceError` telling the user to use `solve()` / `policy=`. PR B fixes this.
11. **`_signature_matches` accepts `**kwargs`-only callables as "matching any parent set"** — a `dist` or `values` that silently ignores its parents is `CONSISTENT`. Documented, but it's a silent-wrongness footgun in the exact layer meant to gate inference.
12. **`UtilityNode.is_discrete` returns `False` as a routing proxy** — semantically "not a BN-engine input", not "not finite". Works today, but conflates two meanings; the first engine that reads `is_discrete` for its own purposes will trip on it.
13. **`infer()` result format seam** — probability vectors for discrete, raw MCMC sample arrays for continuous. One return shape would be nicer; needs a design decision (e.g. always return probability vectors, or always return samples).
14. **`docs/graph/decision_node.md` recommends Strategy B (intervention-scan) for v0** — the actual v0 solver is exact bucket elimination. The docs should be reconciled with what was implemented (Strategy B remains the plan for the NumPyro path / continuous decisions).

## Small nits

15. **`inference/id/bucket_elim.py` elimination loop** — `for name in reversed(snapshot.order): node = node_map[name]` could iterate `reversed(snapshot.nodes)` directly and unpack `(name, node)`, skipping the dict lookup.
16. **`_max_out` in `bucket_elim.py`** — recomputes `base.variables.index(v)` inside the per-context loop; precompute an index map once.
17. **`Factor.__mul__` empty-scope-is-unit-1 semantics** — the trap that caused the 1.0-vs-100.0 bug. `__add__` documents the asymmetry; `__mul__`'s docstring should too, so the next reader doesn't trip on it.
18. **`inference/utils/cpt.py` name** — contains `cardinalities`, `cpt`, and `utility_factor`; the name undersells it. Consider `utils/factors.py` or similar if a rename comes up anyway.
