---
file_format: mystnb
kernelspec:
  name: python3
---

# Common pitfalls

This page collects common mistakes and behaviours when using PyLIMID.

## Indexing a table with several parents

When a node has several discrete parents, chained indexing such as `table[parent_1][parent_2]` can give incorrect posterior results or errors during NumPyro's discrete enumeration. This behaviour is reported in [NumPyro issue #2252](https://github.com/pyro-ppl/numpyro/issues/2252).

To avoid it, we can use both indices in a single operation, `table[parent_1, parent_2]`. For example, if `demand` has discrete parents `weather` and `bookings`:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

DEMAND_MEANS = jnp.array([[40.0, 60.0], [120.0, 180.0]])


def demand_dist(weather, bookings):
    mean = DEMAND_MEANS[weather, bookings]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)
```

## Checking parent effects

Suppose we have two chance nodes, `weather` and `demand`, with `weather` as a parent of `demand`. The demand function uses the weather state's index (`0` for `"cloudy"`, `1` for `"sunny"`) to select a mean from a lookup table.

In the following example, we accidentally include only the cloudy-day mean in that table:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram

diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=("cloudy", "sunny"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("weather",),
        dist=lambda weather: dist.Gamma(
            concentration=4.0,
            rate=4.0 / jnp.array([40.0])[weather], # here
        ),
    )
)

print("validation:", diagram.validate())
for problem in diagram.probe_discrete_parents():
    print(problem.message)
```

{py:meth}`~InfluenceDiagram.validate` reports no problems because it checks argument names, but not how the function uses them. [JAX clamps the out-of-range index](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing) to the cloudy-day value, so both weather states produce the same demand distribution.

{py:meth}`~InfluenceDiagram.probe_discrete_parents` detects that `weather` has no effect. The check runs only when called and reports a warning, since ignoring a parent can be intentional. It does not check whether the values themselves are correct.
