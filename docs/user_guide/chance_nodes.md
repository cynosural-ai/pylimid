---
file_format: mystnb
kernelspec:
  name: python3
---

# Chance nodes

A `ChanceNode` is a random variable, `P(name | parents)`. Its `dist` is a plain callable that receives the resolved parent values as **keyword arguments** — keyed by parent name, not parent order — and returns a NumPyro distribution object. Discrete nodes declare a `states` tuple; omitting `states` (the default `None`) makes the node continuous.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer
```

## Discrete nodes

`states` is a tuple of human-facing labels, and the values flowing through the graph are integer indices into that tuple — `0`, `1`, … in declaration order. Any NumPyro discrete distribution works; a `Categorical` is the common case.

```{code-cell} ipython3
discrete = InfluenceDiagram()
discrete.add_node(
    ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
    )
)
discrete.add_node(
    ChanceNode(
        name="wet_grass",
        parents=("rain",),
        states=("dry", "wet"),
        dist=lambda rain: dist.Categorical(
            probs=jnp.array([[0.9, 0.1], [0.2, 0.8]])[rain]
        ),
    )
)
discrete.validate()
```

A conditional table is just an array indexed by the parent values; `[rain]` picks the row for the realized parent state. Because the dependency is expressed by name, the callable keeps working if the parent order changes.

```{code-cell} ipython3
infer(discrete, ["wet_grass"])["wet_grass"].marginal()
```

Discrete posteriors are exact here: the engine enumerates the discrete sites and bincounts them, so `marginal()` returns the probability vector directly.

## Continuous nodes

Omit `states` and the node's values are real numbers. A conditional callable composes them arithmetically — the classic linear-Gaussian form is a `Normal` whose location is a linear expression in the parents.

```{code-cell} ipython3
continuous = InfluenceDiagram()
continuous.add_node(ChanceNode(name="x1", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
continuous.add_node(
    ChanceNode(
        name="x2",
        parents=("x1",),
        dist=lambda x1: dist.Normal(loc=2.0 + 1.5 * x1, scale=0.5),
    )
)
prior = infer(continuous, ["x1", "x2"])
(round(prior["x1"].mean(), 2), round(prior["x2"].mean(), 2))
```

Continuous variables have no state labels, so their `Posterior` summarizes the draws with `mean()`, `std()`, and `hdi()` rather than `marginal()`. With no observations the engine samples the prior in one pass; observations route through NUTS.

## Mixed and conditional linear-Gaussian

Discrete and continuous nodes combine freely. The conditional linear-Gaussian (CLG) convention — a discrete parent selecting a Gaussian's parameters — is the workhorse mixed pattern:

```{code-cell} ipython3
clg = InfluenceDiagram()
clg.add_node(
    ChanceNode(
        name="quality",
        states=("low", "high"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.4, 0.6])),
    )
)
clg.add_node(
    ChanceNode(
        name="yield",
        parents=("quality",),
        dist=lambda quality: dist.Normal(
            loc=jnp.array([0.0, 3.0])[quality],
            scale=jnp.array([1.0, 0.5])[quality],
        ),
    )
)
round(infer(clg, ["yield"])["yield"].mean(), 2)
```

The marginal of `yield` is a Gaussian mixture, so its prior mean is the mixture mean. Conditioning on a continuous observation sharpens the discrete parent:

```{code-cell} ipython3
infer(clg, ["quality"], observed={"yield": 2.0})["quality"].marginal()
```

## The callable runs inside JAX

`dist` is not called once in Python: it is traced and evaluated inside vectorized forward passes and, for continuous posteriors, inside NUTS. Two consequences:

- Write it as a pure `jnp` expression — no `float()` / `int()` coercion, and no Python `if` on a traced parent value. Use `jnp.where` for branching.
- The signature is the contract. Every parent must be named as a parameter for the node to be `CONSISTENT` (see [](building_a_diagram.md)). A `**kwargs` callable does not count: the gate cannot verify that it reads anything. Callables that resolve parents dynamically should declare a generated named-parameter signature — a common pattern for table-backed distributions.

Editing is the same mutable-workspace story as the rest of the graph: `node.dist = new_callable`, or `diagram.set_dist(name, new_callable)`, and re-validate. Changing the *parent set* makes the node stale until the signature is updated.

## Checking that a parent is used

`validate()` sees the signature, not the body, so a silent indexing bug — a table one row short, where JAX clamps the index and returns the last row — passes validation. `probe_discrete_parents()` runs the callables and varies each discrete parent; if the output never moves, it reports a `DIST_IGNORES_PARENT` warning:

```{code-cell} ipython3
ignores = InfluenceDiagram()
ignores.add_node(
    ChanceNode(
        name="g",
        states=("a", "b"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
    )
)
ignores.add_node(
    ChanceNode(
        name="y",
        parents=("g",),
        states=("no", "yes"),
        dist=lambda g: dist.Categorical(probs=jnp.array([0.5, 0.5])),
    )
)
ignores.probe_discrete_parents()
```

It is a warning, not an error: a node whose distribution genuinely ignores a discrete parent is a legitimate model, and the check cannot tell the two apart. It is also opt-in, because it executes your callables.
