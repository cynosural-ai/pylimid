# Typed per-entry inference results: Marginal / Draws

**Date:** 2026-08-13
**Status:** Superseded by [25_08_2026_unified_posterior_result.md](./25_08_2026_unified_posterior_result.md)

> **Where this fits.** This is a *decision record* — it records the call and what changed in the tree. It is a contract change to `infer()`, whose previous result format was documented and pinned by tests.

---

## The decision

`infer()` returns a dict of typed per-entry results instead of raw lists:

- `Marginal(values, exact)` for discrete query variables — the probability vector over the variable's states, plus whether it is exact (variable elimination) or a Monte-Carlo estimate (numpyro).
- `Draws(values)` for continuous query variables — raw posterior draws.

The old contract `dict[str, list[float]]` was opaque: the caller could not tell a probability vector from a raw sample list without cross-referencing the diagram, and could not tell exact from estimated without knowing which engine auto-dispatch had picked. The typed entries make both distinctions structural — visible to the type checker and to `match` statements.

`ve.query` and `numpyro.samples` are unchanged: their contracts were already honest (exact vectors / raw draws). The engine name remains the quality metadata at the engine level; `Marginal.exact` carries that metadata through `infer()`'s auto-dispatch, where the caller does not name the engine.

## What changes in the tree

| Before | After |
|--------|-------|
| `InferenceResult = dict[str, list[float]]` | `dict[str, Marginal \| Draws]` |
| `result["rain"] == [0.456, 0.544]` | `result["rain"].values`, `result["rain"].exact` |
| `result["temp"]` (raw list, silent) | `result["temp"]` is a `Draws` |

`tests/inference/test_infer.py` pins the new contract: per-variable type, the `exact` flag per engine, and the mixed-diagram shape.

## Rejected alternatives

### Single class with a `kind` tag

A `kind: Literal["marginal", "draws"]` field is runtime-inspectable but does not guide attribute access. Two classes make the contract visible to the type checker and to `match` statements.

### A `Marginal` without the `exact` flag

Removes the shape opacity but leaves the original ambiguity — exact (VE) vs estimated (numpyro) — unresolved for `engine="auto"` callers. The flag is one bool that closes it.

### Removing `infer()`

Pushes engine selection onto every user and loses the normalization path (bincounting draws into vectors). The per-engine entry points already serve users who want to name the engine; `infer()` keeps the ergonomic umbrella with an honest contract.
