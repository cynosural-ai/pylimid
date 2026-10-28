# Chance nodes — distribution specification

Design note covering how a `ChanceNode`'s conditional distribution should be expressed, and the representation chosen as canonical.

> **Scope.** This note settles the *distribution representation* (callable vs. CPT). The node's *mutability model* — editable fields, optional `dist`, `UNCONFIGURED`/`STALE`/`CONSISTENT` states — is a separate concern settled in [`diagram.md`](./diagram.md). The examples below show construction for clarity but the node is mutable in practice: `parents` and `dist` may be set or changed after construction.

---

## The decision

**Canonical form: a callable that takes resolved parent values and returns a NumPyro distribution.** A conditional probability table (CPT) is an array wrapped in such a callable.

```python
# canonical — one form for every case (discrete, continuous, mixed)
ChanceNode(
    name="wet_grass",
    parents=("rain",),
    states=("dry", "wet", "soaked"),
    dist=lambda rain: dist.Categorical(probs=P[rain]),
)
```

The same representation serves tables and other distributions throughout validation, translation and solving.

---

## Context: the fork

A discrete chance node with one discrete parent could be expressed two ways:

- **Callable returning a Distribution** — `lambda rain: dist.Bernoulli(...)`.
- **Conditional probability table (CPT)** — a dict mapping parent-state tuples to probability vectors, in the style of Netica / GeNIe / pgmpy / pyAgrum.

Tables are the tradition in the decision-analysis / operations-research world: explicit, printable, rows that can be checked to sum to 1. But a table is a table over *states*, so it is **discrete-only by construction**. It cannot represent a continuous node (`rainfall ~ Normal(...)`) or a node with a continuous parent. A table-primary design therefore forces continuous variables into a separate node type with a separate code path — which fights the "mixed-type" premise of the library.

---

## Why the callable is canonical

### Asymmetry of the two representations

- **Callable → table is always possible** when the domain is finite: evaluate the callable over every combination of parent states and materialize a table.
- **Table → callable structure is impossible.** Once the functional form is discarded and only the numbers are kept, the parameters that generated it cannot be recovered.

The richer representation should be canonical. The callable does not prevent exact discrete inference (variable elimination, junction tree): a table can always be materialized by evaluating the callable over every parent assignment.

### Uniform translator

The NumPyro translator always emits:

```
fn = node.dist(**resolved_parents)
numpyro.sample(node.name, fn)
```

No branching on "is this a table node or a function node." This matters most in the translator, which is expected to be the most bug-prone layer. One emission path, not two.

### Mixed-type falls out for free

- discrete parent, discrete child → `Categorical(probs=P[parent])`
- continuous parent, continuous child → `Normal(loc=parent, scale=1)`
- mixed → just another lambda

No new node type, no new path. The premise of the library holds.

---

## Discrete parent, discrete child — the CPT case

This is the case where the table tradition is strongest, and where the callable must be shown to not lose anything.

### Values are integer indices, not labels

NumPyro's `Categorical` samples return integer indices (`0, 1, 2, ...`), never the string labels. The value that flows through the graph for a categorical node is an integer. This is a requirement of JAX traceability: a model cannot branch on a string inside a `sample` site.

Labels decode category indices for presentation, while the declaration of `states` also establishes categorical semantics for policy tables and posterior summaries. Strings do not enter the internal sampled-value flow.

### The table is preserved as data

```
           dry   wet  soaked
rain=no  [ 0.80  0.15  0.05 ]
rain=yes [ 0.05  0.35  0.60 ]
```

becomes:

```python
P = jnp.array(
    [
        [0.80, 0.15, 0.05],  # rain=0 (no)
        [0.05, 0.35, 0.60],  # rain=1 (yes)
    ]
)

ChanceNode(
    name="wet_grass",
    parents=("rain",),
    states=("dry", "wet", "soaked"),
    dist=lambda rain: dist.Categorical(probs=P[rain]),
)
```

`P` *is* the CPT — the same table, stored as a matrix and indexed by the parent's integer value. No information loss versus the dict-keyed form:

```
dict-keyed CPT                        matrix form
─────────────────────────             ─────────────────────────
{("no",):  [0.80, 0.15, 0.05],   ⟷   P[0] = [0.80, 0.15, 0.05]
 ("yes",): [0.05, 0.35, 0.60]}        P[1] = [0.05, 0.35, 0.60]
```

For a finite domain there is nothing *but* a table; the callable just wraps "select row, wrap in Categorical."

### Multi-parent generalizes as a higher-rank array

`wet_grass` depending on `rain` and `sprinkler` (both binary):

```python
P = jnp.zeros((2, 2, 3))  # (rain, sprinkler, wet_grass)
# ...fill...
wet_grass_dist = lambda rain, sprinkler: dist.Categorical(probs=P[rain, sprinkler])
```

One axis per parent. No new mechanism.

---

## Role of the `states` field

Since categorical values are integers internally, the label list does real work:

- **Cardinality:** `len(states)` sizes policy-table axes and discrete posterior probability vectors. Distribution outputs must be integer indices from zero to `len(states) - 1`. Structural validation does not execute `dist`, so it cannot establish that runtime contract.
- **Display:** report results as `wet_grass = soaked` rather than `wet_grass = 2`.
- **Decision nodes too:** a discrete decision's action space is its `states` tuple. Policies use integer indices; `Solution.render()` decodes them to labels for display.

`states` is required for categorical chance nodes. Numerical variables, including discrete counts, use `states=None` and retain their actual values. The [categorical and count note](discrete_chance_nodes.md) describes runtime distribution checks and inference limits.
