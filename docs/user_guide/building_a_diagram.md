---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a diagram

This page covers how to build and edit an influence diagram, check that it is complete, and draw it. It also explains how to represent what the owner knows at each decision.

We continue with the ice cream shop from the [Quickstart](../getting_started/quickstart.md). The owner chooses whether to run a promotion the evening before, then decides how much ice cream to order after checking the morning weather. The actual demand and the season's `busyness` remain unknown at both decisions.

## Nodes and arcs

There are three kinds of nodes, and every node has a `name` and a tuple of `parents` (i.e., the nodes where its arcs come from):

- {py:class}`ChanceNode` represents a random variable, like the weather, the demand or the `busyness` multiplier.
- {py:class}`DecisionNode` is a choice, like whether to run the promotion or how much ice cream to order. Its parents are what is known when the choice is made (i.e., its *information set*).
- {py:class}`UtilityNode` indicates the resulting payoff, like the shop's profit. A utility node can't have children: nothing in the diagram depends on a payoff.

We start with the numbers and the `profit` function from the quickstart:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

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


def profit(order, demand, promotion):
    revenue = PRICE * jnp.minimum(order, demand)
    return revenue - COST * order - PROMOTION_COST * promotion
```

## Building in any order

We don't need to fill the whole {py:class}`InfluenceDiagram` in one go. We can add a node before its parents exist, add arcs later, and fill in a node's details as we go.

However, some invalid values still raise an error immediately. PyLIMID checks different kinds of problems at different times:

- When we create or edit a node, PyLIMID rejects invalid values such as an empty name, repeated state labels or an interval whose upper bound is below its lower bound.
- When we call {py:meth}`~InfluenceDiagram.validate`, PyLIMID checks the whole diagram for missing parents, cycles, unconfigured nodes and functions whose arguments don't match their parents.

For example, we can start with the two decisions and `profit`, before adding `weather`, `busyness` or `demand`. The promotion's `states` list its two actions, while the order's `action_space` allows any quantity between zero and the freezer's capacity. A decision declares one of these two kinds of actions. Setting both is a configuration conflict reported by `validate()`.

We keep the same policy families as in the quickstart: {py:class}`TabularSoftmax` for the promotion and {py:class}`DistributionPolicy` with {py:class}`Linear` for the order. The {py:class}`Interval` declares the allowed quantities, and the truncated Normal's bounds keep its orders within that range:

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    DecisionNode(
        name="promotion",
        states=("skip", "run"),
        policy_family=TabularSoftmax(),
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
```

We can draw the diagram at any point, even unfinished. In Jupyter, a diagram on the last line of a cell draws itself with [Graphviz](https://graphviz.org/download/), which needs the `dot` program installed on the system ({py:meth}`~InfluenceDiagram.to_svg` returns the same picture as SVG):

```{code-cell} ipython3
diagram
```

```{admonition} Reading the drawing
:class: note

Chance nodes are circles, decisions are rectangles, and utilities are diamonds. A parent that doesn't exist yet appears as a dashed node, a node that is missing its distribution, actions or utility function is grey, and a stale node is yellow.
```

{py:meth}`~InfluenceDiagram.to_mermaid` returns the same diagram as [Mermaid](https://mermaid.js.org) text, which needs nothing installed and renders in GitHub comments and Markdown files. It is also easier to read for LLMs:

```{code-cell} ipython3
print(diagram.to_mermaid())
```

Now we add the chance nodes. The `weather` and `busyness` nodes are complete, but we leave the `demand` node's distribution out for now:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="weather",
        states=WEATHER_STATES,
        dist=lambda: dist.Categorical(probs=WEATHER_PROBS),
    )
)
diagram.add_node(
    ChanceNode(
        name="busyness",
        dist=lambda: dist.LogNormal(
            jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
        ),
    )
)
diagram.add_node(
    ChanceNode(name="demand", parents=("weather", "promotion", "busyness"))
)
```

## Validating the diagram

{py:meth}`~InfluenceDiagram.validate` checks the whole diagram and returns a list of {py:class}`DiagramProblem`. Each problem has a `kind`, the `node` it belongs to, and a readable `message`:

```{code-cell} ipython3
for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is missing its distribution, so it is marked as `UNCONFIGURED`. To fix this, we can either use {py:meth}`~InfluenceDiagram.set_dist` or assign its distribution directly with `diagram["demand"].dist`:

```{code-cell} ipython3
diagram.set_dist(
    "demand",
    lambda weather, promotion, busyness: dist.Gamma(
        4.0,
        4.0 / (DEMAND_MEANS[weather] * PROMOTION_BOOST[promotion] * busyness),
    ),
)
diagram.validate()
```

An empty list means PyLIMID found no problems with the diagram's structure or node configuration. **This check does not evaluate the distributions or utility function**. Also, we don't have to call `validate()` before solving: {py:func}`solve` and {py:func}`infer` check the diagram themselves and raise an error listing any problems.

The policy families are checked separately by `solve`, which requires one on every decision before training.

Each node also reports its own state through {py:attr}`~Node.consistency`, which is one of `UNCONFIGURED`, `STALE` or `CONSISTENT`:

```{code-cell} ipython3
diagram["demand"].consistency
```

We can now draw the complete diagram:

```{code-cell} ipython3
diagram
```

## What does each decision know?

The two decisions take place at different times. The promotion is chosen the evening before, when the owner does not yet know the weather. The order is chosen in the morning, when the owner knows the weather and remembers whether the promotion is running:

```{code-cell} ipython3
print("Promotion observes:", diagram["promotion"].parents)
print("Order observes:", diagram["order"].parents)
```

Notice that `promotion` is a parent of both `demand` and `order`, but the arcs have different meanings. For `demand`, the parent changes the distribution. For `order`, it says that the owner remembers the earlier choice. The `busyness` multiplier is unknown, so neither decision has it as a parent.

Suppose we remove `promotion` from the parents of `order`:

```{code-cell} ipython3
diagram.remove_arc("promotion", "order")
print("Order observes:", diagram["order"].parents)
print("Validation:", diagram.validate())
diagram
```

The promotion still affects demand and still costs money. But the order decision can now only react to the weather, using the same rule whether the promotion ran or not. **This is a different decision problem**, even though the diagram still passes validation. PyLIMID does not automatically make an earlier choice available to a later decision.

A later decision also needs an explicit parent for each earlier observation it should remember. Making an observation a parent of the first decision does not make its value available to the second one.

We restore the memory arc before continuing with the shop's model:

```{code-cell} ipython3
diagram.add_arc("promotion", "order")
diagram["order"].parents
```

## Changing the diagram

Suppose the owner notices that demand is higher when bookings at nearby hotels are high and wants the model to take it into consideration. However, the owner does not currently have access to the hotels' booking figures, only an estimate based on past records: bookings are normal on roughly five out of seven days and high on the other two. Thus, we add a `bookings` node and an arc from it to `demand`, leaving it unknown at both decisions:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="bookings",
        states=("normal", "high"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_arc("bookings", "demand")

for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is now `STALE`: its distribution is currently a function of `weather`, `promotion` and `busyness`, so it can't use the new parent. The drawing highlights it in yellow:

```{code-cell} ipython3
diagram
```

To fix it, we give `demand` a distribution that considers all four parents. As an example, let's say that high hotel bookings increase average demand by 15%:

```{code-cell} ipython3
BOOKINGS_BOOST = jnp.array([1.0, 1.15])

diagram.set_dist(
    "demand",
    lambda weather, promotion, busyness, bookings: dist.Gamma(
        4.0,
        4.0
        / (
            DEMAND_MEANS[weather]
            * PROMOTION_BOOST[promotion]
            * busyness
            * BOOKINGS_BOOST[bookings]
        ),
    ),
)
diagram.validate()
```

PyLIMID matches a function to its parents by the names of its arguments, so `weather`, `promotion`, `busyness` and `bookings` must be spelled exactly like the parent nodes, in any order:

```{code-cell} ipython3
diagram
```

To remove an arc or node, we can use {py:meth}`~InfluenceDiagram.remove_arc` and {py:meth}`~InfluenceDiagram.remove_node` (which removes a node along with its arcs). These methods do not update the children's functions, so a chance or utility node becomes `STALE` if its function no longer matches its parents.

## Next steps

- [Distributions and utilities](distributions_and_utilities.md) explains the shop's demand assumptions, what functions receive from both kinds of decisions, and how to split profit into revenue and costs.
