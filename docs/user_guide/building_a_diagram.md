---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a diagram

This page covers how to build and edit an influence diagram, check that it is complete, and draw it. What goes inside each node (distributions, decisions and utilities) has its own pages in the user guide.

We use the ice-cream vendor example from the [quickstart](../getting_started/quickstart.md) page as a running example.

## Nodes and arcs

There are three kinds of nodes, and every node has a `name` and a tuple of `parents` (i.e., the nodes where its arcs come from):

- {py:class}`ChanceNode` represents a random variable, like the weather or the demand.
- {py:class}`DecisionNode` is a choice, like how many ice creams to stock. Its parents are what is known when the choice is made (i.e., its *information set*).
- {py:class}`UtilityNode` indicates the resulting payoff, like the vendor's profit. A utility node can't have children: nothing in the diagram depends on a payoff.

We start with the numbers and the profit function from the quickstart:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

DEMAND_MEANS = jnp.array([40.0, 120.0])
ORDERS = jnp.arange(0, 201, 20)


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked
```

## Building in any order

You don't need to fill the whole {py:class}`InfluenceDiagram` in one go. You can add a node before its parents exist, add arcs later, and fill in a node's details whenever you are ready. Nothing is checked until you ask for it, not even cycles: {py:meth}`~InfluenceDiagram.validate` reports every problem at once.

For example, we can start with the decision and the profit, before describing the weather or the demand:

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
```

The diagram can be drawn at any point, even unfinished. In Jupyter, a diagram on the last line of a cell draws itself with [Graphviz](https://graphviz.org/download/), which needs the `dot` program installed on your system ({py:meth}`~InfluenceDiagram.to_svg` returns the same picture as SVG):

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


Now we add the two chance nodes. The weather node is complete, but we leave the demand node's distribution out for now:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="weather",
        states=("cloudy", "sunny"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
diagram.add_node(ChanceNode(name="demand", parents=("weather",)))
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
    "demand", lambda weather: dist.Gamma(4.0, 4.0 / DEMAND_MEANS[weather])
)
diagram.validate()
```

An empty list means the diagram is complete and ready to be solved. You don't have to call `validate()` before solving: {py:func}`solve` and {py:func}`infer` check the diagram themselves and raise an error listing the problems if it isn't complete.

Each node also reports its own state through {py:attr}`~Node.consistency`, which is one of `UNCONFIGURED`, `STALE` or `CONSISTENT`:

```{code-cell} ipython3
diagram["demand"].consistency
```

## Changing the diagram

Suppose the vendor notices that weekends are busier than weekdays and wants the model to take it into consideration. Thus, we add a `weekend` node and an arc from it to `demand`:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="weekend",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_arc("weekend", "demand")

for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is now `STALE`: its distribution is currently a function of `weather` only, so it can't use the new parent. The drawing highlights it in yellow:

```{code-cell} ipython3
diagram
```

To fix it, we give `demand` a distribution that considers both parents. As an example, let's say that weekends multiply the average demand by 1.5:

```{code-cell} ipython3
WEEKEND_BOOST = jnp.array([1.0, 1.5])

diagram.set_dist(
    "demand",
    lambda weather, weekend: dist.Gamma(
        4.0, 4.0 / (DEMAND_MEANS[weather] * WEEKEND_BOOST[weekend])
    ),
)
diagram.validate()
```

PyLIMID matches a function to its parents by the names of its arguments, so `weather` and `weekend` must be spelled exactly like the parent nodes, in any order.

To take things out of the diagram in running time, you can use {py:meth}`~InfluenceDiagram.remove_arc` and {py:meth}`~InfluenceDiagram.remove_node` (which removes a node along with its arcs). Removing doesn't update the children's functions, so they would become `STALE`.
