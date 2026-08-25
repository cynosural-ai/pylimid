# Unified inference result: Posterior

**Date:** 2026-08-25
**Status:** Settled

> **Where this fits.** This is a *decision record* — it records the call and what changed in the tree. It replaces `ADR/13_08_2026_typed_inference_results.md` (marked superseded). It is a contract change to `infer()`, whose per-entry result format was pinned by tests.

---

## The decision

`infer()` returns one `Posterior` per query variable instead of two type-level shapes:- `Posterior(values, states)` — raw posterior draws plus the variable's state labels (`None` for continuous).
- `marginal()` — the discrete probability vector (bincounted draws). Raises `ValueError` on a continuous posterior: a continuous variable has no per-state probability mass, and a loud error beats a silent wrong answer.
- `mean()`, `std()`, `hdi(prob=0.95)` — continuous summary methods. Raise `ValueError` on a discrete posterior: the mean of state indices depends on their encoding, so it would be silently misleading for nominal states — the mirror of the `marginal()` guard.

Every method is valid for exactly one variable kind and raises on the other; the discreteness discriminator is explicit and queryable (`states is not None` / `is_discrete`), and the raw draws remain available for a caller who deliberately wants an index-based statistic.

This replaces `Marginal(values)` (probability vector, discrete only) and `Draws(values)` (raw draws, continuous only).

## Why

Draws are the universal representation: a discrete variable's draws are integer state indices, and the probability vector is a bincount of them — a lossless summary. A continuous variable has no such aggregate: mean±std is a projection that only fully describes a Gaussian, a histogram is an arbitrary binning choice, and the raw draws are the one representation from which every summary can be recovered. Two result types made the engine pick the *representation* silently by variable type; one type keeps the draws always and exposes the summaries as methods, with the variable's discreteness (`states is not None`) as the explicit, queryable discriminator. The `marginal()` guard is the one method whose meaning is type-dependent, so it is the one method that raises.

## What changes in the tree

| Before | After |
|--------|-------|
| `dict[str, Marginal \| Draws]` | `dict[str, Posterior]` |
| `result["rain"].values` (probability vector) | `result["rain"].marginal()` |
| `result["temp"].values` (draws) | `result["temp"].values` (draws) + `mean()`/`std()`/`hdi()` |
| engine bincounts discrete draws and discards them | engine keeps draws; `marginal()` bincounts on demand |

`tests/inference/test_infer.py` pins the new contract: one `Posterior` per query, `states` present or `None` by variable type, draws stored as raw state indices for discrete nodes, and the `marginal()` `ValueError` on continuous posteriors.

## Rejected alternatives

### Two types, shared base / protocol

Keeps `Marginal` and `Draws` but gives them common summary methods. Preserves type-level distinction at the cost of two near-identical classes that the engine must still dispatch on; the single type moves the discreteness question into the result's own `states` field instead.

### Mean and std as the continuous "marginal"

A 2-parameter summary only characterizes a Gaussian posterior; a bimodal, skewed, or mixture posterior can share mean±std while looking nothing alike. As the stored representation it would be silently misleading whenever the posterior is not Gaussian — and `infer()` cannot know the family from an arbitrary `dist` callable. Valid as a derived convenience (`mean()`/`std()`), never as the result's shape.
