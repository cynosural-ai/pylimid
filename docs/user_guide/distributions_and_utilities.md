---
file_format: mystnb
kernelspec:
  name: python3
---

# Distributions and utilities

In [Building a diagram](building_a_diagram.md), we assigned a distribution to each chance node and a utility function to the `profit` node. But why those distributions in particular? Could we, for example, assume that demand is Gaussian instead of {py:class}`Gamma <numpyro.distributions.continuous.Gamma>`? What would happen if we split the utility function into revenue and cost?

This page answers those questions using the same ice cream shop example. It also explains what these functions receive from their parents and how to write them so PyLIMID can run them with JAX.

## What do the functions receive?

PyLIMID calls each function with one keyword argument per parent, **matched by name**. For `demand`, the discrete parents `weather` and `bookings` pass state indices, while the continuous parent `busyness` passes a numerical multiplier:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])


def demand_dist(weather, busyness, bookings):
    mean = DEMAND_MEANS[weather] * busyness * BOOKINGS_BOOST[bookings]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)
```

For `weather`, `0` means `"cloudy"` and `1` means `"sunny"`. For `bookings`, they mean `"normal"` and `"high"`. The function uses those indices to look up the demand mean and the booking multiplier, then returns a {py:class}`NumPyro distribution <numpyro.distributions.distribution.Distribution>`.

The `profit` function receives two parent values:

- `order` is a **decision node**, so it passes the index of the chosen action to the function. That is why `profit` uses `ORDERS[order]` to get the number stocked.
- `demand` is a **continuous chance node**, so the function receives its value, which is the number of ice creams customers would like to buy.

```{code-cell} ipython3
ORDERS = jnp.arange(0, 201, 20)


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked
```

Thus, a chance node's function returns a distribution, while a utility node's function returns a number. Both can receive indices and numbers, rather than the labels shown in the diagram. The argument names must match the parents, but their order in the function does not matter. The distribution function for a node without parents, like `weather`, takes no arguments.

## Choosing a distribution

A chance node with `states` defined uses {py:func}`Categorical <numpyro.distributions.discrete.Categorical>` with one label per category, including those with probability zero, or {py:func}`Bernoulli <numpyro.distributions.discrete.Bernoulli>` with exactly two labels. Both accept probabilities (`probs`) or logits (`logits`). Draws are the indices `0`, `1`, ... of those states.

```{code-cell} ipython3
def weather_dist():
    return dist.Categorical(probs=jnp.array([0.7, 0.3]))


def bookings_dist():
    return dist.Categorical(probs=jnp.array([5 / 7, 2 / 7]))
```

The probabilities follow the same order as the state labels: `"cloudy"` and `"sunny"` for weather, `"normal"` and `"high"` for bookings. Neither node has parents, so these functions take no arguments.

With `states=None`, use a continuous distribution. The `busyness` node uses the {py:class}`LogNormal <numpyro.distributions.continuous.LogNormal>` prior:

```{code-cell} ipython3
def busyness_dist():
    return dist.LogNormal(
        jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
    )
```

```{admonition} Count distributions
:class: info
Count distributions such as {py:class}`Poisson <numpyro.distributions.discrete.Poisson>` and {py:func}`Binomial <numpyro.distributions.discrete.Binomial>` are not currently supported.
```

We chose a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` distribution to model `demand` because it is strictly positive and can naturally model quiet days alongside some unusually busy ones. Even though ice creams are counted in whole numbers, we use a continuous distribution to approximate demand. We use `concentration=4.0`, so for fixed parent values its standard deviation is half its mean.

### What if demand were Gaussian?

Suppose the owner believes that demand is distributed more symmetrically around its mean. A Gaussian distribution could express that assumption, but it could also produce negative values. Since demand cannot be negative, we use NumPyro's {py:class}`TruncatedNormal <numpyro.distributions.truncated.TruncatedNormal>` with a lower bound of zero:

```{code-cell} ipython3
def nonnegative_demand(weather, busyness, bookings):
    mean = DEMAND_MEANS[weather] * busyness * BOOKINGS_BOOST[bookings]
    return dist.TruncatedNormal(loc=mean, scale=mean / 2, low=0.0)
```

Here `loc` and `scale` are the mean and standard deviation of the Gaussian before truncation. We set `loc` to the mean used in `demand_dist` and `scale` to half of it. Restricting the distribution to nonnegative values raises its actual mean above `loc`, so the two distributions do not have exactly the same average demand.

We can compare both distributions for a sunny day with normal hotel bookings, fixing `busyness` at its prior mean of 1.2. The Gamma curve has a longer tail towards high demand, while the truncated Gaussian is more symmetric around its peak:

```{figure} /_static/figures/distributions_and_utilities/demand_comparison.png
:alt: Gamma and truncated Gaussian demand densities for sunny weather, normal hotel bookings and busyness fixed at 1.2. The Gamma curve has a longer right tail.
:width: 90%
```

We keep the original Gamma demand for the utility examples below. NumPyro has [other distributions](https://num.pyro.ai/en/stable/distributions.html) when a different assumption fits the problem better.

## Splitting the payoff into parts

We can now put together the diagram using the functions defined above:

```{code-cell} ipython3
from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(name="weather", states=("cloudy", "sunny"), dist=weather_dist)
)
diagram.add_node(ChanceNode(name="busyness", dist=busyness_dist))
diagram.add_node(
    ChanceNode(name="bookings", states=("normal", "high"), dist=bookings_dist)
)
diagram.add_node(
    ChanceNode(
        name="demand", parents=("weather", "busyness", "bookings"), dist=demand_dist
    )
)
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

diagram
```

The shop's `profit` function combines revenue and stock cost. Revenue depends on the order size and demand, while cost only depends on how much the owner orders. We can put each part in its own {py:class}`UtilityNode`. We make a copy of the diagram and replace `profit` with these two nodes so we can compare both versions:

```{code-cell} ipython3
from copy import deepcopy

split_diagram = deepcopy(diagram)
split_diagram.remove_node("profit")
split_diagram.add_node(
    UtilityNode(
        name="revenue",
        parents=("order", "demand"),
        values=lambda order, demand: 10.0 * jnp.minimum(ORDERS[order], demand),
    )
)
split_diagram.add_node(
    UtilityNode(
        name="cost", parents=("order",), values=lambda order: -1.0 * ORDERS[order]
    )
)

split_diagram
```

The cost is negative because it reduces what the shop earns. When there are several utility nodes, {py:func}`solve` **adds their values on each simulated day and maximises the expected total**. We can check that the two ways of expressing profit give the same result:

```{code-cell} ipython3
combined_solution = solve(diagram, num_samples=50_000)
split_solution = solve(split_diagram, num_samples=50_000)
print("same policy:", combined_solution.policy == split_solution.policy)
print(f"combined expected profit: {combined_solution.expected_utility:.1f}")
print(f"split expected profit: {split_solution.expected_utility:.1f}")
```

The recommended orders are the same, as is the estimated expected profit. We use the same random key, the default, for both calls. The next guide, [Solving](solving.md), gives more information about {py:func}`solve` and explains how to read this result.

```{admonition} Limitations
:class: info

PyLIMID combines utility nodes by adding their values. If the payoff does not split into a sum, we need a single utility function with all the relevant parents.

For example, a nonlinear function expressing the owner's attitude to risk should be applied to the total profit, rather than separately to revenue and cost.
```

## Writing the functions for JAX

PyLIMID evaluates these functions using [JAX](https://docs.jax.dev/en/latest/), where parent values are represented as JAX arrays. The functions must be **pure**, i.e., compute their result from their arguments and constants **without changing the diagram or other state while they run**.

As a general tip, keep lookup tables such as `DEMAND_MEANS` and `ORDERS` as JAX arrays. Standard arithmetic operators (`+`, `-`, `/`, `*`) work with these arrays. To take a minimum or express a condition, we use `jnp`.

The following utility function contains several operations that would fail when JAX compiles it:

```python
def incorrect_profit(order, demand):
    # For casting, use jnp.asarray(order, dtype=jnp.int32).
    stocked = ORDERS[int(order)]
    # For casting, use jnp.asarray(demand, dtype=jnp.float32).
    wanted = float(demand)
    sold = min(stocked, wanted)  # Use jnp.minimum(stocked, demand).

    if demand > stocked:  # Use jnp.where(demand > stocked, -50.0, 0.0).
        penalty = -50.0
    else:
        penalty = 0.0

    return 10.0 * sold - stocked + penalty
```

```{admonition} Understanding JAX compilation
:class: info

JAX's [Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html) guide explains how functions are traced and compiled, with several examples of pure functions.
```
