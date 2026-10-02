# Unified inference layer: two verbs, auto-dispatch

**Date:** 2026-07-23
**Status:** Superseded in part by [`25_08_2026_numpyro_only_engine.md`](./25_08_2026_numpyro_only_engine.md) — the unified package and the two verbs (`infer`, `solve`) remain; the auto-dispatch and the multi-engine table below do not describe the current library (NumPyro is the only engine).
**Supersedes:** the single-backend `backend/numpyro.py` as the sole inference surface.

> **Where this fits.** This is a *decision record* — it records the call and what will change in the tree. The *reasoning* lives in [`inference_strategy.md`](../inference_strategy.md); this note documents the architectural choices.

---

## The decision

The library exposes **two public verbs** for model execution:

```python
from decisionpy.inference import infer, solve

result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
policy = solve(diagram)
```

`infer` computes posterior distributions over named variables. `solve` computes optimal decision policies and expected utilities. `infer` defaults to the NumPyro engine and dispatches to the exact engines (`ve`, `lg`) only on an explicit `engine=` request; `solve` still picks its engine automatically.

The `backend/` package is collapsed into `inference/`. All algorithms — PPL bridges and graph-native methods alike — live under one roof. The directory distinction between "backend" (PPL) and "inference" (graph-native) was an artifact of having only one algorithm; it no longer holds.

### Engines and their constraints

| Engine | Module | Works on | Dependency | Availability |
|--------|--------|----------|------------|--------------|
| `"ve"` | `variable_elim.py` | BNs only (chance nodes) | `numpy` | Always |
| `"bucket_elim"` | `bucket_elim.py` | Full influence diagrams | `numpy` | Always |
| `"gibbs"` | `gibbs.py` | Any diagram | `numpy` | Always |
| `"numpyro"` | `numpyro.py` | Any diagram | `numpyro` extra | Optional |

### Target directory layout

```
src/decisionpy/inference/
├── __init__.py          # exports infer(), solve()
├── _dispatch.py         # inspect diagram → choose engine
├── _numpyro_model.py    # internal: to_model (Snap → NumPyro model)
├── numpyro.py           # mcmc(), svi(), forward_sample()
├── variable_elim.py     # query() — exact BN marginals
├── bucket_elim.py       # solve() — exact ID policies
├── arc_reversal.py      # internal helper for bucket_elim
└── gibbs.py             # gibbs() — approximate, any diagram
```

The existing `src/decisionpy/backend/` directory is removed.

### Policy parameter: decisions in `infer()`

`infer()` accepts an optional `policy` argument:

```python
result = infer(diagram, query=["disease"], observed={"symptom": 1}, policy={"treat": 0})
```

Binding a decision clamps it to a fixed value — the node behaves like observed evidence. Once **all** decisions are bound (either via `policy=` or from a previous `solve()` call), utilities are ignored and the diagram collapses to a de facto Bayesian network. At that point the dispatch logic routes to standard inference engines (VE, Gibbs, NumPyro) regardless of whether the diagram originally had decisions.

This enables interventional / counterfactual queries without calling `solve()`:

```python
# Hypothesis test: P(recovery | treat=0, symptom) vs P(recovery | treat=1, symptom)
no_treatment = infer(diagram, query=["recovery"], observed={...}, policy={"treat": 0})
with_treatment = infer(diagram, query=["recovery"], observed={...}, policy={"treat": 1})
```

If any decision remains unbound, `infer()` rejects:

```
InferenceError: diagram has unbound decision nodes: ['test'].
Specify policy={"test": ...} or call solve() first.
Currently bound: treat=0.
```

The check is `set(decisions) - set(policy.keys())`. All-or-nothing — partially bound is still ambiguous. `solve()` is the only path that fills in missing policy entries automatically (by optimizing expected utility).

For NumPyro's `to_model`, bound decisions are injected as observed sites alongside regular observations — no special-case code path needed.

### Engine selection rules

```
infer(diagram, engine="numpyro"):            # the default
    the NumPyro backend handles any diagram (Monte-Carlo; exact for
    enumerated discrete variables only in expectation — results are
    never flagged exact).

infer(diagram, engine="ve"):                 # explicit opt-in
    exact discrete Bayesian-network marginals. Requires an all-discrete
    diagram (fails loudly otherwise).

infer(diagram, engine="lg"):                 # explicit opt-in
    exact linear-Gaussian marginals. Requires an all-continuous diagram
    of Normal CPDs with affine means (fails loudly otherwise).

solve(diagram, engine="bucket_elim"):        # the default
    exact policies for all-categorical diagrams. The NumPyro
    intervention-scan solver is planned; engine="numpyro" raises
    InferenceError until it lands.
```

The exact engines are explicit opt-ins: each validates its own preconditions and rejects diagrams that do not satisfy them. There is no inference-time dispatch between them — the general engine is the default, exactness is a deliberate choice.

### Explicit engine errors

When a user picks an engine that cannot handle their diagram:

```
InferenceError: variable_elimination requires a Bayesian network
(chance nodes only). This diagram has decision nodes: ['invest'].
Use engine="bucket_elim", "gibbs", or "numpyro".
```

No silent fallback — the user made a deliberate choice and gets a deliberate explanation of why it doesn't work and what alternatives are available.

### Why not a registry / plugin system

The engines are few, their constraints are structural (node types, graph size, discrete vs mixed), and the dispatch logic is simple. A registry would be over-engineering for what is really one `if/elif` block. If the engine list grows beyond ~5 or third-party engines become a thing, a registry can be added as a follow-up ADR.

### Relationship to `graph/`

`graph/` imports nothing from `inference/`. The dependency arrow remains one-way: graph → nothing. This is the same guarantee the old `backend/` package provided, now enforced by the `inference/` directory boundary.

---

## What changes in the tree

| Before | After |
|--------|-------|
| `src/decisionpy/backend/__init__.py` | Removed |
| `src/decisionpy/backend/numpyro.py` | `src/decisionpy/inference/_numpyro_model.py` (internal) + `numpyro.py` (public) |
| No inference API | `infer()`, `solve()` in `inference/__init__.py` |
| `tests/test_backend_numpyro.py` | Split into `tests/inference/test_numpyro.py` and dispatch tests |
| — | `src/decisionpy/inference/variable_elim.py` |
| — | `src/decisionpy/inference/gibbs.py` |
| — | `src/decisionpy/inference/bucket_elim.py`, `arc_reversal.py` |
| `pyproject.toml` `numpyro` extra | Unchanged (same optional deps) |

## Status of dependent decisions

| Topic | Status |
|-------|--------|
| Inference strategy (algorithm choice) | Settled in `inference_strategy.md` |
| Directory layout | Settled here |
| `infer()` signature (query, observed, policy) | Settled here |
| `solve()` signature | Settled here |
| Policy binding collapses ID → BN | Settled here |
| Auto-dispatch rules | Settled here |
| Exact variable elimination | Planned — next step |
| Gibbs sampling | Planned |
| Bucket elimination / arc reversal | Planned — after VE |

---

## Rejected alternatives

### Keep `backend/` + separate `inference/`

Adds a distinction ("backend" vs "inference") that only made sense when there was one algorithm. Users wouldn't know which directory to import from, and internal re-exports would create circular confusion.

### PPL-first: NumPyro as the only engine

Locks the library into a heavy dependency for work that doesn't need it (exact BN inference, discrete-only diagrams). Forces every user to install JAX even for a 4-node Bayesian network. Contradicts the zero-dep goal of the graph layer.

### Graph-native only: no PPL at all

Abandons gradient-based decision optimization for large/continuous decision spaces, GPU acceleration, and the entire Pyro/NumPyro ecosystem. The `inference_strategy.md` analysis settled on NumPyro as the primary engine for mixed-type diagrams at scale; the graph-native methods serve exact discrete inference and as a lightweight alternative.
