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
prior-predictive sampling). That is the v0 capability.

---

## Dependency

NumPyro (and its JAX base) is a declared runtime dependency of decisionpy, not
a separate extra. The graph layer still never imports it — `decisionpy.graph`
works on plain `dist` callables, and `import decisionpy` does not load NumPyro;
only importing `decisionpy.inference` does. `to_model` raises a clear
`ImportError` if NumPyro cannot be imported.

---

## v0 scope, and what is deferred

**In scope:**
- Chance nodes only (a Bayesian network).
- Forward / prior-predictive sampling, via `Predictive`.

**Deliberately deferred:**
- **Posterior inference** (NUTS / SVI over continuous latents given
  observations). Forward sampling is what decision solving (Strategy B in
  [`decision_node.md`](./decision_node.md)) builds on; posterior inference is a
  later capability with real algorithm choices to commit to.
- **Decision and utility nodes.** The translator remains chance-node only
  (it narrows every node to `ChanceNode`); a diagram with decisions or
  utilities routes to `solve()` / the `id` solvers, not here.
- **A deep-frozen snapshot.** A `Snapshot` holds references to still-mutable
  nodes ([`diagram.md`](./diagram.md), "the snapshot is logical, not
  deep-frozen"). The node list is captured once at `to_model` time, so
  structural edits afterward don't affect an already-captured model; but field
  mutation on a captured node *before* sampling would be seen by the model.
  Freezing is the documented future tightening; v0 assumes the caller does not
  mutate captured nodes.
