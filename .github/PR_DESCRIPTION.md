<!-- 
General notes:

 * Dont break lines at 80 characters, let lines flow and the viewer will adjust 
-->

## Summary

Adds an exact influence-diagram solver via **bucket elimination** (`decisionpy.inference.id.solve`), which computes optimal per-decision policies and maximum expected utility for all-categorical LIMIDs. It shares the factor machinery with variable elimination by promoting the `Factor` primitive and CPT builder into shared modules.

## What changed

**New shared primitives**
- `decisionpy/inference/_factor.py` — the `Factor` class, moved from `ve/factor.py` (history preserved via git). Used by both VE and the ID solver; `id/` no longer imports from `ve/`.
- `decisionpy/inference/_cpt.py` — new module with `cardinalities()` (name → domain size), `cpt()` (a chance node's CPT factor, promoted from `ve/elim.py`), and `utility_factor()` (a utility node's `U(parents)` factor).

**New solver — `decisionpy/inference/id/`**
- `bucket_elim.py` — `solve(snapshot) -> Solution` where `Solution` is a dataclass holding `policy` (decision → `{info_set_assignment: action}`) and `expected_utility`. Algorithm: probability factors multiply, utility factors combine additively (`E[sum U_k]`), variables are eliminated in reverse topological order (chance nodes fold into the utility factor via `sum_X P·U`; decisions are maxed out per information-set assignment).
- `__init__.py` — public API surface (`solve`).

**Refactor — `decisionpy/inference/ve/elim.py`**
- Uses the shared `cpt()`/`cardinalities()` builders; removes the duplicated `_cpt`/`_extract_probs` helpers; fixes a stale docstring reference.

**Dependency**
- `pyagrum>=1.12` added to the `test` dependency group (`pyproject.toml`, `uv.lock`).

**Tests**
- `tests/inference/test_bucket_elim.py` — 6 unit tests (hand-computed): single decision (EU 78), two independent decisions with additive utility (EU 90), empty information set, irrelevant information, continuous-node `TypeError`, and non-regular LIMID `NotImplementedError`.
- `tests/validation/_id_fixtures.py` + `tests/validation/test_vs_pyagrum.py` — cross-validation against pyAgrum's exact LIMID solver (`ShaferShenoyLIMIDInference`): 6 fixtures (incl. nested two-decision and 3-state domains), asserting MEU and per-decision optimal policies agree to `abs=1e-5`.

**Docs**
- `docs/README.md` — code-status update (ID solver now implemented; `solve()` wiring in `engine.py` moved to planned).
- `docs/sessions/SESSION_2026_08_09.md` — session summary with algorithm details and correctness subtleties.

## Why

This is the first `solve()` milestone for influence diagrams: an exact, graph-native, numpy-only counterpart to variable elimination that produces optimal policies (the current docs only planned it). It enables posterior-intervention reasoning over the diagram without sampling. Cross-validation against pyAgrum's exact solver establishes correctness on a range of structures.

## Testing

- `uv run pytest tests/` — 170 passed (was 164).
- `uv run ruff check .` and `uv run ruff format --check .` — clean.
- `uv run ty check` — clean.
- 6 hand-computed unit tests plus 6 pyAgrum cross-validation tests. No solver bugs found during validation; only test-harness issues (pyAgrum v3 API differences, two fixture arc omissions, float32 rounding looseness).
