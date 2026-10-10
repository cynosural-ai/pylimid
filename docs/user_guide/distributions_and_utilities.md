---
file_format: mystnb
kernelspec:
  name: python3
---

# Distributions and utilities

Every chance node needs a distribution, and every utility node needs a utility function. In PyLIMID both are Python functions of the node's parents. This page covers how to write them: what they receive, what they return, the rules that come from running them inside JAX, and a few modelling patterns.

We continue with the ice-cream vendor from the [quickstart](../getting_started/quickstart.md) page.

## Functions of the parents

Here is the vendor's model again: the weather, the demand it drives, how many ice creams to stock, and the profit.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

DEMAND_MEANS = jnp.array([40.0, 120.0])
ORDERS = jnp.arange(0, 201, 20)


def demand_dist(weather):
    return dist.Gamma(concentration=4.0, rate=4.0 / DEMAND_MEANS[weather])


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked


diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=("cloudy", "sunny"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
diagram.add_node(ChanceNode(name="demand", parents=("weather",), dist=demand_dist))
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
```

PyLIMID calls each function with one keyword argument per parent, matched by name, so `demand_dist` must have an argument called `weather` and `profit` must have `order` and `demand`, in any order. A node without parents, like `weather`, takes no arguments.

What each argument holds depends on the parent:

- A **discrete chance node** passes the index of its state: `0` for cloudy and `1` for sunny. That is why `demand_dist` can use `weather` to pick an entry of `DEMAND_MEANS`.
- A **decision node** passes the index of the chosen action, not the action itself. That is why `profit` looks up `ORDERS[order]` to get the number of ice creams stocked.
- A **continuous chance node** passes its value, a real number like `demand` above.

A chance node's function returns a [NumPyro distribution](https://num.pyro.ai/en/stable/distributions.html), and a utility node's function returns a number.

## Discrete and continuous chance nodes

A chance node with `states` is discrete. Its distribution must produce the indices `0`, `1`, … of those states, which is what {py:func}`Categorical <numpyro.distributions.discrete.Categorical>` does. When a discrete node has discrete parents, its probabilities are usually a table with one row per combination of parent states.

For example, suppose the vendor only sees a forecast, which is right 80% of the time on cloudy days and 90% on sunny ones. Each row of the table is the forecast's distribution for one kind of weather, and `[weather]` picks the row:

```{code-cell} ipython3
FORECAST_PROBS = jnp.array(
    [
        [0.8, 0.2],  # cloudy: forecast says cloudy, sunny
        [0.1, 0.9],  # sunny
    ]
)

diagram.add_node(
    ChanceNode(
        name="forecast",
        parents=("weather",),
        states=("cloudy", "sunny"),
        dist=lambda weather: dist.Categorical(probs=FORECAST_PROBS[weather]),
    )
)
```

A chance node without `states` is continuous, like `demand`. Any continuous NumPyro distribution works, here a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>`, and its parameters can depend on the parents: `demand_dist` picks the mean demand for the day's weather. A continuous parent arrives as a number, so you can use it directly in arithmetic, for example as the mean of a {py:class}`Normal <numpyro.distributions.continuous.Normal>`. Chance nodes can be discrete, continuous, or both in the same model.

## Utility functions

A utility function returns the payoff for one combination of parent values. When a diagram has several utility nodes, PyLIMID adds them up, and {py:func}`solve` maximises the expected total. This lets you split a payoff into parts that depend on different nodes. Here we replace `profit` with a revenue, which depends on the order and the demand, and a cost, which depends only on the order:

```{code-cell} ipython3
diagram.remove_node("profit")
diagram.add_node(
    UtilityNode(
        name="revenue",
        parents=("order", "demand"),
        values=lambda order, demand: 10.0 * jnp.minimum(ORDERS[order], demand),
    )
)
diagram.add_node(
    UtilityNode(
        name="cost", parents=("order",), values=lambda order: -1.0 * ORDERS[order]
    )
)

solution = solve(diagram, num_samples=50_000)
for info, action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[info[0]]}: stock {int(ORDERS[action])}")
print(f"expected profit: {float(solution.expected_utility):.1f}")
```

The policy is the same as in the quickstart, where a single `profit` node held the whole payoff: stock 60 ice creams on a cloudy day and 200 on a sunny one, for an expected profit of about 507. The forecast node doesn't change it, because the order still depends on the weather itself.

Adding up is the only way PyLIMID combines utility nodes. To weight one part more than another, multiply inside its function. When parts don't simply add up, for example when the vendor cares about the worse of two outcomes, use a single utility node with all the relevant parents.

## Risk attitude

Maximising expected profit treats a sure \$500 the same as a coin flip between \$0 and \$1,000. A vendor who can't afford a bad day may prefer the sure \$500. Decision analysis captures this with a concave utility function, which grows more slowly as profit increases, so a loss hurts more than an equal gain helps (see Clemen, *Making Hard Decisions*, on risk attitudes). A common choice is the exponential utility \(1 - e^{-\text{profit}/R}\), where the risk tolerance \(R\) is in dollars and a smaller \(R\) means a more cautious vendor.

Because a utility function is just a function of its parents, a risk-averse vendor only needs a different utility node:

```{code-cell} ipython3
RISK_TOLERANCE = 200.0


def satisfaction(order, demand):
    return 1.0 - jnp.exp(-profit(order, demand) / RISK_TOLERANCE)


diagram.remove_node("revenue")
diagram.remove_node("cost")
diagram.add_node(
    UtilityNode(name="satisfaction", parents=("order", "demand"), values=satisfaction)
)

solution = solve(diagram, num_samples=50_000)
for info, action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[info[0]]}: stock {int(ORDERS[action])}")
print(f"expected utility: {float(solution.expected_utility):.2f}")
```

The cautious vendor still stocks 60 ice creams on a cloudy day, but only 100 on a sunny one instead of 200. Sunny days are busy but unpredictable, and a large stock loses money whenever a sunny day turns out quiet. The expected utility is now on the utility function's scale, between 0 and 1, rather than in dollars.

## Written for JAX

PyLIMID evaluates these functions inside [JAX](https://docs.jax.dev/en/latest/), many times over, with the parent values given as JAX arrays rather than plain Python numbers. This is what lets solving and inference run efficiently on CPU and GPU, and it means each function must be written as a JAX expression:

- Use `jnp` functions (`jnp.minimum`, `jnp.exp`, …) for arithmetic.
- Don't convert parent values with `float()` or `int()`.
- Don't branch on a parent value with a Python `if`; JAX raises a `TracerBoolConversionError`. Use `jnp.where` instead.

For example, a penalty of \$50 whenever the vendor runs out of ice cream is written with `jnp.where`:

```{code-cell} ipython3
def stockout_penalty(order, demand):
    return jnp.where(demand > ORDERS[order], -50.0, 0.0)
```

The arguments must also be named after the parents. A function that takes `**kwargs` doesn't count, because PyLIMID can't tell which parents it reads, so its node stays `STALE` until the function names every parent.

## Checking that a discrete parent is used

{py:meth}`~InfluenceDiagram.validate` checks a function's arguments, not what it does with them. A common silent mistake is a probability table with a missing row. Suppose the forecast table had only its cloudy row:

```{code-cell} ipython3
diagram["forecast"].dist = lambda weather: dist.Categorical(
    probs=jnp.array([[0.8, 0.2]])[weather]
)
diagram.validate()
```

`validate()` is happy, because the function takes `weather`. But JAX doesn't raise an error for an index past the end of an array: it uses the last row instead. So on sunny days the forecast silently uses the cloudy row, and the weather has no effect on it. {py:meth}`~InfluenceDiagram.probe_discrete_parents` catches this. It runs each function once for every state of each discrete parent, and reports a parent that never changes the result:

```{code-cell} ipython3
for problem in diagram.probe_discrete_parents():
    print(problem.kind.name, "-", problem.message)
```

It reports a warning rather than an error, because a node that genuinely doesn't depend on one of its parents looks the same. It only runs when you call it, since it executes your functions.
