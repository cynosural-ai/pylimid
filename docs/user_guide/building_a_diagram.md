---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a diagram

This page covers how to build and edit an influence diagram, check that it is complete, and draw it. What goes inside each node (distributions, decisions and utilities) has its own pages in the user guide.

We use the ice cream shop example from the [Quickstart](../getting_started/quickstart.md) page as a running example.

## Nodes and arcs

There are three kinds of nodes, and every node has a `name` and a tuple of `parents` (i.e., the nodes where its arcs come from):

- {py:class}`ChanceNode` represents a random variable, like the weather, the demand or the `busyness` multiplier.
- {py:class}`DecisionNode` is a choice, like how many ice creams to stock. Its parents are what is known when the choice is made (i.e., its *information set*).
- {py:class}`UtilityNode` indicates the resulting payoff, like the shop's profit. A utility node can't have children: nothing in the diagram depends on a payoff.

```{admonition} What a decision knows
:class: info

A decision observes exactly its declared parents. In the shop model, `weather` is a parent of `order`, so the owner knows the weather before ordering. Demand and the actual `busyness` are unknown, so neither is a parent of `order`.

PyLIMID does not automatically make earlier choices or observations available to later decisions. Anything a later decision should remember must also be one of its parents.

Decisions choose from a finite set of actions and can only observe discrete variables.
```

We start with the numbers and the `profit` function from the quickstart:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
ORDERS = jnp.arange(0, 201, 20)


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked
```

## Building in any order

We don't need to fill the whole {py:class}`InfluenceDiagram` in one go. We can add a node before its parents exist, add arcs later, and fill in a node's details as we go.

However, some invalid values still raise an error immediately. PyLIMID checks different kinds of problems at different times:

- When we create or edit a node, PyLIMID rejects invalid values such as an empty name or repeated state labels.
- When we call {py:meth}`~InfluenceDiagram.validate`, PyLIMID checks the whole diagram for missing parents, cycles, unconfigured nodes and functions whose arguments don't match their parents.

For example, we can start with the `order` and `profit` nodes, before adding `weather`, `busyness` or `demand`:

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
```

We can draw the diagram at any point, even unfinished. In Jupyter, a diagram on the last line of a cell draws itself with [Graphviz](https://graphviz.org/download/), which needs the `dot` program installed on the system ({py:meth}`~InfluenceDiagram.to_svg` returns the same picture as SVG):

```{code-cell} ipython3
diagram
```

```{admonition} Reading the drawing
:class: note

Chance nodes are circles, decision nodes are rectangles, and utility nodes are diamonds. A parent that doesn't exist yet appears as a dashed node, a node that is missing its distribution, actions or utility function is grey, and a stale node is yellow.
```

{py:meth}`~InfluenceDiagram.to_mermaid` returns the same diagram as [Mermaid](https://mermaid.js.org) text, which needs nothing installed and renders in GitHub comments and Markdown files. It is also "easier to read" for LLMs:

```{code-cell} ipython3
print(diagram.to_mermaid())
```

Now we add the chance nodes. The `weather` and `busyness` nodes are complete, but we leave the `demand` node's distribution out for now:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="weather",
        states=("cloudy", "sunny"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
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
diagram.add_node(ChanceNode(name="demand", parents=("weather", "busyness")))
```

## Validating the diagram

{py:meth}`~InfluenceDiagram.validate` checks the whole diagram and returns a list of {py:class}`DiagramProblem`. Each problem has a `kind`, the `node` it belongs to, and a readable `message`.

```{code-cell} ipython3
for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is missing its distribution, so it is marked as `UNCONFIGURED`. To fix this, we can either use {py:meth}`~InfluenceDiagram.set_dist` or assign its distribution directly with `diagram["demand"].dist`.

```{code-cell} ipython3
diagram.set_dist(
    "demand",
    lambda weather, busyness: dist.Gamma(
        4.0, 4.0 / (DEMAND_MEANS[weather] * busyness)
    ),
)
diagram.validate()
```

An empty list means PyLIMID found no problems with the diagram's structure or node configuration. **This check does not evaluate the distributions or utility functions**. Also, we don't have to call `validate()` before solving: {py:func}`solve` and {py:func}`infer` check the diagram themselves and raise an error listing any problems.

Each node also reports its own state through {py:attr}`~Node.consistency`, which is one of `UNCONFIGURED`, `STALE` or `CONSISTENT`:

```{code-cell} ipython3
diagram["demand"].consistency
```

## Changing the diagram

Suppose the owner notices that demand is higher when bookings at nearby hotels are high and wants the model to take it into consideration. However, the owner does not currently have access to the hotels' booking figures, only an estimate based on past records: bookings are normal on roughly five out of seven days and high on the other two. Thus, we add a `bookings` node and an arc from it to `demand` but not to `order`:

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

`demand` is now `STALE`: its distribution is currently a function of `weather` and `busyness`, so it can't use the new parent. The drawing highlights it in yellow:

```{code-cell} ipython3
diagram
```

To fix it, we give `demand` a distribution that considers all three parents. As an example, let's say that high hotel bookings increase average demand by 15%:

```{code-cell} ipython3
BOOKINGS_BOOST = jnp.array([1.0, 1.15])

diagram.set_dist(
    "demand",
    lambda weather, busyness, bookings: dist.Gamma(
        4.0, 4.0 / (DEMAND_MEANS[weather] * busyness * BOOKINGS_BOOST[bookings])
    ),
)
diagram.validate()
```

PyLIMID matches a function to its parents by the names of its arguments, so `weather`, `busyness` and `bookings` must be spelled exactly like the parent nodes, in any order.

```{code-cell} ipython3
diagram
```

To remove an arc or node, we can use {py:meth}`~InfluenceDiagram.remove_arc` and {py:meth}`~InfluenceDiagram.remove_node` (which removes a node along with its arcs). These methods do not update the children's functions, so a chance or utility node becomes `STALE` if its function no longer matches its parents.
