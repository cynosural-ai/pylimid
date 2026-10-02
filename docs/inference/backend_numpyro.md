# NumPyro engine — translator, samplers, solver

The bridge between the backend-agnostic graph layer and a concrete probabilistic programming system. This is the most bug-prone layer (per [`diagram.md`](./diagram.md)), so it is kept deliberately thin.

> **Where this fits.** The graph layer ([`diagram.md`](./diagram.md)) settles the container and the `dist` calling convention; the translator consumes a validated `Snapshot` and turns it into a runnable model. A chance-node-only diagram *is* a Bayesian network; an influence diagram is translated once every decision is bound by a policy (the `policy=` binding on `infer()`, see `decisionpy.inference.engine`).

---

## What it does

`to_model(snapshot, observed) -> model` walks the snapshot's nodes in topological order (parents before children — guaranteed by the snapshot) and emits, per node:

- **chance node** — one `numpyro.sample` with parents resolved **by name**:

  ```python
  fn = node.dist(**{parent: sampled_value_of(parent) for parent in node.parents})
  value = numpyro.sample(node.name, fn)
  ```

- **bound decision** — a degenerate observed site at its policy value (`infer(policy=...)` merges the policy into `observed`; unbound decisions never reach the translator).
- **utility node** — skipped entirely (a payoff is never sampled and is never a parent).

The translator does *translation only*; how the returned `model` is then sampled or inferred is the caller's concern:

```python
from numpyro.infer import Predictive

model = to_model(diagram.snapshot())
prior = Predictive(model, num_samples=10_000)(jax.random.PRNGKey(0))
```

`Predictive` with no posterior samples draws from the **prior** (forward / prior-predictive sampling). The public `samples()` entry point wraps this: `samples(snapshot)` forward-samples, and `samples(snapshot, observed=...)` runs posterior inference and returns one JAX array per node.

**Indexing in `dist` callables.** Under parallel enumeration the parent values a `dist` callable receives may carry NumPyro's enumeration dimension. Index multi-parent probability tables in a single indexing operation (`T[a, b]`), not chained (`T[a][b]`): NumPyro computes silently wrong posterior weights for chained indexing. This reproduces in pure NumPyro (a two-node model with an enumerated parent), so it is a NumPyro-side behavior the translator documents rather than fixes. Open upstream issue: <https://github.com/pyro-ppl/numpyro/issues/2252> (revisit this note when it is resolved).

---

## Dependency

NumPyro (and its JAX base) is a declared runtime dependency of decisionpy, not a separate extra. The graph layer still never imports it — `decisionpy.graph` works on plain `dist` callables, and `import decisionpy` does not load NumPyro; only importing `decisionpy.inference` does.

---

## Scope

**Handled:**
- Chance nodes — the Bayesian-network case.
- Bound decisions — clamped by a policy, emitted as degenerate observed sites.
- Utility nodes — skipped.
- Forward / prior-predictive sampling: `samples(snapshot)` with no `observed` — direct, independent draws.
- Posterior inference: `samples(snapshot, observed=...)`. When every latent node is discrete, the posterior is drawn by exact discrete enumeration in one vectorized pass; otherwise NUTS samples the continuous latents (with the discrete ones enumerated out) and a conditional pass resamples the discrete latents given the continuous draws. The returned dict covers every node; observed nodes appear as their clamped values.

**Not handled here:**
- **Unbound decisions.** The translator does not check binding itself — an unbound decision fails with a `KeyError`, and `infer()` rejects unbound diagrams with a clean `InferenceError` before any engine runs.
- **A deep-frozen snapshot.** A `Snapshot` holds references to still-mutable nodes ([`diagram.md`](./diagram.md), "the snapshot is logical, not deep-frozen"). The node list is captured once at `to_model` time, so structural edits afterward don't affect an already-captured model; but field mutation on a captured node *before* sampling would be seen by the model. Freezing is the documented future tightening; v0 assumes the caller does not mutate captured nodes.

---

## The solver — intervention scan

`numpyro/solvers/intervention_scan.py` solves influence diagrams by Strategy B ([`decision_node.md`](../graph/decision_node.md)): it does not go through `to_model`. It enumerates the discrete policy space — one action per information-set assignment per decision — and evaluates each policy by a forward simulation that mirrors the translator's topological walk, except that a decision node is not a sample site: its action is looked up from the candidate policy as a deterministic function of the realized information-set values (`jax.vmap` over the samples). Utility nodes are evaluated on the sampled values after the walk; their average over the samples is the expected-utility estimate, and the argmax policy is returned as a `Solution`.

Scope and cost, per the strategy analysis in `decision_node.md`:

- **Discrete decisions with discrete information sets** — a continuous information set cannot be tabulated and raises a clean `InferenceError`; that is the policy-as-parameters path (Strategy A), deferred.
- **Exponential in the policy space** — the product over decisions of actions-per-information-set-assignment; forward passes per policy are the Monte-Carlo evaluations. Suits small decision spaces.

The implementation is cross-validated against pyAgrum's exact LIMID solver (`tests/inference/numpyro/test_compare_pyagrum_limid.py`).

Two more solvers live beside it in `numpyro/solvers/`:

- `batched_scan.py` — the same scan with every policy evaluated in one vmapped, jitted pass; same policy and expected utility, a fraction of the runtime.
- `backward_induction.py` — resolves the decisions in reverse order from the utilities, estimating each action's continuation value per information-set assignment by stratified forward sampling (trajectories grouped by the realized assignment). Additive in the decisions instead of exponential in the policy space. It requires a soluble diagram — the regularity gate is `numpyro/solvers/regularity.py`, implementing the exact-solution-ordering criterion of Lauritzen and Nilsson (stricter than pyAgrum's level-based `isSolvable()`; see [`solver_algorithms.md`](./solver_algorithms.md)); a non-soluble diagram raises. Validated against both pyAgrum's exact solver and the scan (`tests/inference/numpyro/solvers/test_backward_induction.py`).
