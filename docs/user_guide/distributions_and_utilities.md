---
file_format: mystnb
kernelspec:
  name: python3
---

# Distributions and utilities

In [Building a diagram](building_a_diagram.md), we assigned a distribution to each chance node and a utility function to the `profit` node. But why those distributions in particular? Could we, for example, assume that demand is Gaussian instead of {py:class}`Gamma <numpyro.distributions.continuous.Gamma>`? What would happen if we split the utility function into revenue and costs?

This page answers those questions using the same ice cream shop example, including the hotel bookings we added to demand. It also explains what the functions receive from their parents and how to write them so PyLIMID can run them with JAX.

## What do the functions receive?

PyLIMID calls each function with one keyword argument per parent, **matched by name**. For `demand`, the categorical parents `weather`, `promotion` and `bookings` pass state indices, while the continuous parent `busyness` passes a numerical multiplier:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

COST = 10.0
PRICE = 100.0
CAPACITY = 40.0
PROMOTION_COST = 60.0
PROMOTION_BOOST = jnp.array([1.0, 1.2])
WEATHER_STATES = ("cloudy", "sunny")
WEATHER_PROBS = jnp.array([0.7, 0.3])
DEMAND_MEANS = jnp.array([4.0, 12.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])


def demand_dist(weather, promotion, busyness, bookings):
    mean = (
        DEMAND_MEANS[weather]
        * PROMOTION_BOOST[promotion]
        * busyness
        * BOOKINGS_BOOST[bookings]
    )
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)
```

For `weather`, `0` means `cloudy` and `1` means `sunny`. For `promotion`, they mean `skip` and `run`, and for `bookings`, `normal` and `high`. The function uses these indices to look up the demand mean and the two multipliers, then returns a {py:class}`NumPyro distribution <numpyro.distributions.distribution.Distribution>`.

The `profit` function receives three parent values:

- `order` is a **continuous decision**, so it passes the amount ordered in kilograms.
- `demand` is a **continuous chance variable**, so it passes the quantity customers would like to buy, also in kilograms.
- `promotion` is a **categorical decision**, so it passes the index of the chosen action. Here, multiplying by `0` skips the promotion cost and multiplying by `1` charges it.

```{code-cell} ipython3
def profit(order, demand, promotion):
    revenue = PRICE * jnp.minimum(order, demand)
    return revenue - COST * order - PROMOTION_COST * promotion
```

Thus, a chance node's function returns a distribution, while a utility node's function returns a number. Both can receive indices and numerical values, rather than the labels shown in the diagram. What a decision passes depends on whether its actions are categorical or continuous.

The argument names must match the parents, but their order in the function does not matter. The distribution function for a node without parents, like `weather`, takes no arguments.

## Choosing a distribution

A chance node with `states` defined uses {py:func}`Categorical <numpyro.distributions.discrete.Categorical>` with one label per category, including those with probability zero, or {py:func}`Bernoulli <numpyro.distributions.discrete.Bernoulli>` with exactly two labels. Both accept probabilities (`probs`) or logits (`logits`). Draws are the indices `0`, `1`, ... of those states:

```{code-cell} ipython3
def weather_dist():
    return dist.Categorical(probs=WEATHER_PROBS)


def bookings_dist():
    return dist.Categorical(probs=jnp.array([5 / 7, 2 / 7]))
```

The probabilities follow the same order as the state labels: `cloudy` and `sunny` for weather, `normal` and `high` for bookings. Neither node has parents, so these functions take no arguments.

With `states=None`, values are numerical. We can use a continuous distribution or a [count distribution](../resources/count_distributions.md). The `busyness` node uses the {py:class}`LogNormal <numpyro.distributions.continuous.LogNormal>` prior:

```{code-cell} ipython3
def busyness_dist():
    return dist.LogNormal(
        jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
    )
```

We chose a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` distribution to model `demand` because it is strictly positive and can naturally model quiet days alongside some unusually busy ones. Demand is measured by weight, so it can take continuous values. We use `concentration=4.0`, so for fixed parent values its standard deviation is half its mean.

### What if demand were Gaussian?

Suppose the owner believes that demand is distributed more symmetrically around its mean. A Gaussian distribution could express that assumption, but it could also produce negative values. Since demand cannot be negative, we use NumPyro's {py:class}`TruncatedNormal <numpyro.distributions.truncated.TruncatedNormal>` with a lower bound of zero:

```{code-cell} ipython3
def nonnegative_demand(weather, promotion, busyness, bookings):
    mean = (
        DEMAND_MEANS[weather]
        * PROMOTION_BOOST[promotion]
        * busyness
        * BOOKINGS_BOOST[bookings]
    )
    return dist.TruncatedNormal(loc=mean, scale=mean / 2, low=0.0)
```

Here `loc` and `scale` are the mean and standard deviation of the Gaussian before truncation. We set `loc` to the mean used in `demand_dist` and `scale` to half of it. Restricting the distribution to nonnegative values raises its actual mean above `loc`, so the two distributions do not have exactly the same average demand.

We can compare both distributions for a sunny day when the promotion runs and hotel bookings are normal, fixing `busyness` at its prior mean of 1.2. The Gamma curve has a longer tail towards high demand, while the truncated Gaussian is more symmetric around its peak:

```{figure} /_static/figures/distributions_and_utilities/demand_comparison.png
:alt: Gamma and truncated Gaussian demand densities in kilograms for sunny weather, a running promotion, normal hotel bookings and busyness fixed at 1.2. The Gamma curve has a longer right tail.
:width: 90%
```

Notice that we only set a lower bound for demand. Customers may want more than the freezer can hold, so demand can exceed 40 kilograms. The capacity limits what the owner can order, not what customers would like to buy.

We keep the original Gamma demand for the utility examples below. NumPyro has [other distributions](https://num.pyro.ai/en/stable/distributions.html) when a different assumption fits the problem better.

### What about the order distribution?

The demand distribution expresses the owner's uncertainty about how much customers would like to buy. The distribution attached to `order` through its policy family describes how the owner chooses stock. Its location is learned when we fit the policies, using the information available at that decision.

We keep the order's truncated Normal policy from the quickstart, with fixed bounds of zero and 40 kilograms and a fixed scale of 0.25 kilograms. Changing the demand distribution changes our assumptions about customers. Changing the order's policy family changes the rules we allow the solver to fit.

## Splitting the payoff into parts

We can now put together the diagram using the functions defined above. We keep the same two decisions and their information: the promotion is chosen the evening before, and the order observes the weather and remembers the promotion. Hotel bookings and `busyness` remain unknown at both decisions:

```{code-cell} ipython3
from pylimid import (
    ChanceNode,
    DecisionNode,
    DistributionPolicy,
    InfluenceDiagram,
    Interval,
    Linear,
    TabularSoftmax,
    UtilityNode,
)

diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(name="weather", states=WEATHER_STATES, dist=weather_dist)
)
diagram.add_node(
    DecisionNode(
        name="promotion",
        states=("skip", "run"),
        policy_family=TabularSoftmax(),
    )
)
diagram.add_node(ChanceNode(name="busyness", dist=busyness_dist))
diagram.add_node(
    ChanceNode(name="bookings", states=("normal", "high"), dist=bookings_dist)
)
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("weather", "promotion", "busyness", "bookings"),
        dist=demand_dist,
    )
)
diagram.add_node(
    DecisionNode(
        name="order",
        parents=("weather", "promotion"),
        action_space=Interval(0.0, CAPACITY),
        policy_family=DistributionPolicy(
            distribution=dist.TruncatedNormal,
            parameter_model=Linear(outputs=("loc",)),
            fixed_parameters={"low": 0.0, "high": CAPACITY, "scale": 0.25},
        ),
    )
)
diagram.add_node(
    UtilityNode(
        name="profit", parents=("order", "demand", "promotion"), values=profit
    )
)

diagram
```

The shop's `profit` function combines revenue, stock cost and promotion cost. Revenue depends on the amount ordered and the demand. Stock cost depends only on how much the owner orders, while promotion cost depends only on whether the promotion runs.

We can put each part in its own {py:class}`UtilityNode`. We remove `profit` and replace it with three nodes:

```{code-cell} ipython3
def revenue(order, demand):
    return PRICE * jnp.minimum(order, demand)


def stock_cost(order):
    return -COST * order


def promotion_cost(promotion):
    return -PROMOTION_COST * promotion


diagram.remove_node("profit")
diagram.add_node(
    UtilityNode(name="revenue", parents=("order", "demand"), values=revenue)
)
diagram.add_node(
    UtilityNode(name="stock_cost", parents=("order",), values=stock_cost)
)
diagram.add_node(
    UtilityNode(name="promotion_cost", parents=("promotion",), values=promotion_cost)
)

diagram
```

The costs are negative because they reduce what the shop earns. When there are several utility nodes, {py:func}`solve` **adds their values on each simulated day and maximises the expected total**. We remove the original `profit` node so the same payoff is not counted twice.

For example, suppose the promotion runs, the owner orders 8.3 kilograms and customers buy 6. We can check the two ways of calculating that day's profit:

```{code-cell} ipython3
ordered = jnp.array(8.3)
wanted = jnp.array(6.0)
promoted = jnp.array(1)

print("Revenue:", revenue(ordered, wanted))
print("Stock cost:", stock_cost(ordered))
print("Promotion cost:", promotion_cost(promoted))
print("Combined profit:", profit(ordered, wanted, promoted))
print(
    "Sum of utility values:",
    revenue(ordered, wanted) + stock_cost(ordered) + promotion_cost(promoted),
)
```

The two calculations give the same profit of \$457. The stock cost includes everything ordered, even the ice cream that goes unsold. The promotion cost is charged whenever the promotion runs, regardless of how much the shop sells.

This equality holds for every combination of order, demand and promotion, so the split model has the same expected-profit objective and the same available policies as before. We keep the three utility nodes for the rest of the example.

```{admonition} Combining utilities
:class: info

PyLIMID combines utility nodes by adding their values. If the payoff does not split into a sum, we need a single utility function with all the relevant parents.

For example, a nonlinear function expressing the owner's attitude to risk should be applied to the total profit, rather than separately to revenue and costs.
```

## Writing the functions for JAX

PyLIMID evaluates these functions using [JAX](https://docs.jax.dev/en/latest/), where parent values are represented as JAX arrays. The functions must be **pure**, i.e., compute their result from their arguments and constants **without changing the diagram or other state while they run**.

As a general tip, keep lookup tables such as `DEMAND_MEANS` and `PROMOTION_BOOST` as JAX arrays. Standard arithmetic operators (`+`, `-`, `/`, `*`) work with these arrays. To take a minimum or express a condition, we use `jnp`.

The following utility function adds a penalty when the shop runs out of stock, but contains several operations that would fail when JAX compiles it:

```python
def incorrect_profit(order, demand, promotion):
    stocked = float(order)  # Use order directly.
    promotion_index = int(promotion)  # Use promotion directly.
    sold = min(stocked, demand)  # Use jnp.minimum(order, demand).

    if demand > stocked:  # Use jnp.where(demand > order, -50.0, 0.0).
        penalty = -50.0
    else:
        penalty = 0.0

    return PRICE * sold - COST * stocked - PROMOTION_COST * promotion_index + penalty
```

There is no need to convert the parent values to Python numbers inside these functions. If we need to change an array's type, we can use `jnp.asarray(value, dtype=...)`.

We can check that our original `profit` function compiles with {py:func}`jax.jit`:

```{code-cell} ipython3
compiled_profit = jax.jit(profit)
compiled_profit(order=ordered, demand=wanted, promotion=promoted)
```

```{admonition} Understanding JAX compilation
:class: info

JAX's [Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html) guide explains how functions are traced and compiled, with several examples of pure functions.
```

## Next steps

- [Solving a diagram](solving.md) fits the shop's two policies, explains how to read their results, and checks how training and the order family affect what we find.
