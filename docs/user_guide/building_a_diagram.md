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

You don't need to fill the whole {py:class}`InfluenceDiagram` in one go. You can add a node before its parents exist, add arcs later, and fill in a node's details whenever you are ready. Completeness isn't checked until you ask for it.

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

The diagram can be drawn at any point, even unfinished. Parents that don't exist yet appear as dashed nodes:

```{code-cell} ipython3
diagram
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

## Changing the model

Suppose the vendor notices that weekends are busier than weekdays. We add a `weekend` node and an arc from it to `demand`:

```{code-cell} ipython3
diagram.add_node(
    ChanceNode(
        name="weekend",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_edge("weekend", "demand")

for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is now `STALE`: its distribution is a function of `weather` only, so it can't use the new parent, and PyLIMID doesn't guess what the arc should mean. The drawing highlights it in yellow:

```{code-cell} ipython3
diagram
```

To fix it, we give `demand` a distribution that takes both parents. Weekends multiply the average demand by 1.5:

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

Unlike {py:meth}`~InfluenceDiagram.add_node`, {py:meth}`~InfluenceDiagram.add_edge` is checked straight away: both nodes must already exist, and an arc that would create a cycle or leave a utility node is rejected with an error. To take things out, {py:meth}`~InfluenceDiagram.remove_edge` removes an arc and {py:meth}`~InfluenceDiagram.remove_node` removes a node along with its arcs. Removing doesn't update the children's functions, so a child that used the removed node becomes stale until you update its distribution.

## Drawing the diagram

In Jupyter, a diagram on the last line of a cell draws itself, as in the examples above. Chance nodes are circles, decision nodes are rectangles, and utility nodes are diamonds. Drawing never validates the diagram, so it also shows what is left to do: a missing parent appears as a dashed node, an unconfigured node is grey, and a stale node is yellow.

Drawing needs the [Graphviz](https://graphviz.org/download/) `dot` program installed on your system; without it, Jupyter shows a one-line text summary instead. {py:meth}`~InfluenceDiagram.to_svg` returns the picture as SVG, and {py:meth}`~InfluenceDiagram.to_mermaid` returns it as [Mermaid](https://mermaid.js.org) text, which needs nothing installed and is handy in GitHub comments or prompts to an LLM.
