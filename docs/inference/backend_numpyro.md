# NumPyro backend — chance-node translator

The bridge between the backend-agnostic graph layer and a concrete
probabilistic programming system. This is the most bug-prone layer (per
[`diagram.md`](./diagram.md)), so it is kept deliberately thin.

> **Where this fits.** The graph layer ([`diagram.md`](./diagram.md)) settles
> the container and the `dist` calling convention; the translator consumes a
> validated `Snapshot` and turns it into a runnable model. A chance-node-only
> diagram *is* a Bayesian network, so v0 is a BN translator.

---

## What it does

`to_model(snapshot) -> model` walks the snapshot's nodes in topological order
(parents before children — guaranteed by the snapshot) and emits one
`numpyro.sample` per node, with parents resolved **by name**:

```python
fn = node.dist(**{parent: sampled_value_of(parent) for parent in node.parents})
value = numpyro.sample(node.name, fn)
```

This is exactly the per-node emission pattern fixed by
[`diagram.md`](./diagram.md) — one path, no branching on node sub-type. The
translator does *translation only*; how the returned `model` is then sampled or
inferred is the caller's concern:

```python
from numpyro.infer import Predictive

model = to_model(diagram.snapshot())
prior = Predictive(model, num_samples=10_000)(jax.random.PRNGKey(0))
```

`Predictive` with no posterior samples draws from the **prior** (forward /
prior-predictive sampling). The public `samples()` entry point wraps this:
`samples(snapshot)` forward-samples, and `samples(snapshot, observed=...)`
runs posterior inference and returns one JAX array per node.

---

## Dependency

NumPyro (and its JAX base) is a declared runtime dependency of decisionpy, not
a separate extra. The graph layer still never imports it — `decisionpy.graph`
works on plain `dist` callables, and `import decisionpy` does not load NumPyro;
only importing `decisionpy.inference` does.

---

## v0 scope, and what is deferred

**In scope:**
- Chance nodes only (a Bayesian network).
- Forward / prior-predictive sampling: `samples(snapshot)` with no
  `observed` — direct, independent draws.
- Posterior inference: `samples(snapshot, observed=...)`. When every latent
  node is discrete, the posterior is drawn by exact discrete enumeration in
  one vectorized pass; otherwise NUTS samples the continuous latents (with
  the discrete ones enumerated out) and a conditional pass resamples the
  discrete latents given the continuous draws. The returned dict covers
  every node; observed nodes appear as their clamped values.

**Deliberately deferred:**
- **Decision and utility nodes.** The translator remains chance-node only
  (it narrows every node to `ChanceNode`); `infer()` rejects diagrams with
  decision or utility nodes at the API level and points at `solve()` /
  `policy=`, so they never reach this translator.
- **A deep-frozen snapshot.** A `Snapshot` holds references to still-mutable
  nodes ([`diagram.md`](./diagram.md), "the snapshot is logical, not
  deep-frozen"). The node list is captured once at `to_model` time, so
  structural edits afterward don't affect an already-captured model; but field
  mutation on a captured node *before* sampling would be seen by the model.
  Freezing is the documented future tightening; v0 assumes the caller does not
  mutate captured nodes.
