# NumPyro backend — snapshot translator

The bridge between the backend-agnostic graph layer and a concrete
probabilistic programming system. This is the most bug-prone layer (per
[`diagram.md`](./diagram.md)), so it is kept deliberately thin.

> **Where this fits.** The graph layer ([`diagram.md`](./diagram.md)) settles
> the container and the `dist` calling convention; the translator consumes a
> validated `Snapshot` and turns it into a runnable model. A chance-node-only
> diagram *is* a Bayesian network; an influence diagram is translated once
> every decision is bound by a policy (see the `policy=` binding in
> [`bucket_elim.md`](./bucket_elim.md)).

---

## What it does

`to_model(snapshot, observed) -> model` walks the snapshot's nodes in
topological order (parents before children — guaranteed by the snapshot) and
emits, per node:

- **chance node** — one `numpyro.sample` with parents resolved **by name**:

  ```python
  fn = node.dist(**{parent: sampled_value_of(parent) for parent in node.parents})
  value = numpyro.sample(node.name, fn)
  ```

- **bound decision** — a degenerate observed site at its policy value
  (`infer(policy=...)` merges the policy into `observed`; unbound decisions
  never reach the translator).
- **utility node** — skipped entirely (a payoff is never sampled and is never
  a parent).

The translator does *translation only*; how the returned `model` is then
sampled or inferred is the caller's concern:

```python
from numpyro.infer import Predictive

model = to_model(diagram.snapshot())
prior = Predictive(model, num_samples=10_000)(jax.random.PRNGKey(0))
```

`Predictive` with no posterior samples draws from the **prior** (forward /
prior-predictive sampling). The public `samples()` entry point wraps this:
`samples(snapshot)` forward-samples, and `samples(snapshot, observed=...)`
runs posterior inference and returns one JAX array per node.

**Indexing in `dist` callables.** Under parallel enumeration the parent values
a `dist` callable receives may carry NumPyro's enumeration dimension. Index
multi-parent probability tables in a single indexing operation
(`T[a, b]`), not chained (`T[a][b]`): NumPyro computes silently wrong
posterior weights for chained indexing. This reproduces in pure NumPyro (a
two-node model with an enumerated parent), so it is a NumPyro-side behavior
the translator documents rather than fixes. Open upstream issue:
<https://github.com/pyro-ppl/numpyro/issues/2252> (revisit this note when it
is resolved).

---

## Dependency

NumPyro (and its JAX base) is a declared runtime dependency of decisionpy, not
a separate extra. The graph layer still never imports it — `decisionpy.graph`
works on plain `dist` callables, and `import decisionpy` does not load NumPyro;
only importing `decisionpy.inference` does.

---

## Scope

**Handled:**
- Chance nodes — the Bayesian-network case.
- Bound decisions — clamped by a policy, emitted as degenerate observed sites.
- Utility nodes — skipped.
- Forward / prior-predictive sampling: `samples(snapshot)` with no
  `observed` — direct, independent draws.
- Posterior inference: `samples(snapshot, observed=...)`. When every latent
  node is discrete, the posterior is drawn by exact discrete enumeration in
  one vectorized pass; otherwise NUTS samples the continuous latents (with
  the discrete ones enumerated out) and a conditional pass resamples the
  discrete latents given the continuous draws. The returned dict covers
  every node; observed nodes appear as their clamped values.

**Not handled here:**
- **Unbound decisions.** The translator does not check binding itself — an
  unbound decision fails with a `KeyError`, and `infer()` rejects unbound
  diagrams with a clean `InferenceError` before any engine runs.
- **A deep-frozen snapshot.** A `Snapshot` holds references to still-mutable
  nodes ([`diagram.md`](./diagram.md), "the snapshot is logical, not
  deep-frozen"). The node list is captured once at `to_model` time, so
  structural edits afterward don't affect an already-captured model; but field
  mutation on a captured node *before* sampling would be seen by the model.
  Freezing is the documented future tightening; v0 assumes the caller does not
  mutate captured nodes.
