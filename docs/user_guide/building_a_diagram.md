---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a diagram

This page covers how to build and edit an influence diagram, check that it is complete, and draw it. What goes inside each node (distributions, decisions and utilities) has its own pages in the user guide.

We use the ice-cream vendor example from the [quickstart](../getting_started/quickstart.md) page to guide

## Nodes and arcs

There are three kinds of nodes, and every node has a `name` and a tuple of `parents` (i.e., the nodes where its arcs come from):

- `ChanceNode` represents a random variable, like the weather or the demand.
- `DecisionNode` is a choice, like how many ice creams to stock. Its parents are what is known when the choice is made (i.e., its *information set*).
- `UtilityNode` indicate the resulting payoff, like the vendor's profit. A utility node can't have children: nothing in the diagram depends on a payoff.

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

You don't need to fill the whole `InfluenceDiagram` in one go. You can add a node before its parents exist, add arcs later, and fill in a node's details whenever you are ready. Nothing is checked until you ask for it.

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

Now we add the two chance nodes. The weather node is complete, but we leave demand node's distribution out for now:

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

`validate()` checks the whole diagram and returns a list of `DiagramProblem`. Each problem has a `kind`, the `node` it belongs to, and a readable `message`. 

```{code-cell} ipython3
for problem in diagram.validate():
    print(problem.kind.name, "-", problem.message)
```

`demand` is missing its distribution, so it is marked as `UNCONFIGURED`. To fix this, we can either use `set_dist` or assign its distribution directly with `diagram["demand"].dist`.

```{code-cell} ipython3
diagram.set_dist(
    "demand", lambda weather: dist.Gamma(4.0, 4.0 / DEMAND_MEANS[weather])
)
diagram.validate()
```

An empty list means the diagram is complete and ready to be solved.

Each node also reports its own state through `consistency`, which is one of `UNCONFIGURED`, `STALE` or `CONSISTENT`:

```{code-cell} ipython3
diagram["demand"].consistency
```

---------

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

`demand` is now stale. Its distribution is a function of `weather` only, so it can't use the new parent, and PyLIMID doesn't guess what the arc should mean. The drawing highlights the stale node:

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

PyLIMID matches a function to its parents by the names of its arguments, so `weather` and `weekend` must be spelled exactly like the parent nodes. Their order doesn't matter.

Unlike `add_node`, `add_edge` checks its arguments straight away. Both nodes must already exist, an arc out of a utility node is rejected, and so is any arc that would create a loop:

```{code-cell} ipython3
:tags: [raises-exception]

diagram.add_edge("demand", "weekend")
```

Adding an arc that already exists does nothing. To take things out, `remove_edge(parent, child)` removes an arc, and `remove_node(name)` removes a node along with the arcs that leave it. Removing doesn't update the children's functions, though: if we removed `weekend` now, `demand`'s distribution would still expect it, and `demand` would be stale until we changed it back.

## Looking at the structure

A diagram can be inspected at any time. `names` lists the nodes in the order they were added, and `topological_sort()` orders them so that every node comes after its parents:

```{code-cell} ipython3
diagram.names
```

```{code-cell} ipython3
diagram.topological_sort()
```

`parents_of` and `children_of` follow the arcs in each direction:

```{code-cell} ipython3
diagram.parents_of("demand"), diagram.children_of("weather")
```

A diagram also behaves like a collection of nodes: `diagram["demand"]` returns a node, `"weekend" in diagram` checks whether a node exists, and `len(diagram)` counts them.

## Drawing the diagram

A diagram can be drawn in two formats, which always show the same picture. Chance nodes are circles, decision nodes are rectangles, and utility nodes are diamonds.

- **SVG**, for notebooks and documents. In Jupyter, a diagram on the last line of a cell draws itself, as in the examples above, and `to_svg()` returns the SVG source. Drawing needs the [Graphviz](https://graphviz.org/download/) `dot` program installed on your system. Without it, Jupyter falls back to a one-line text summary, and `to_svg()` raises an error.
- **Mermaid**, a text format. `to_mermaid()` needs nothing installed and returns the same output every time for the same diagram, which makes it useful in diffs, logs, GitHub comments, and prompts to an LLM:

```{code-cell} ipython3
print(diagram.to_mermaid())
```

Drawing never validates the diagram, so it is useful while building one. A missing parent appears as a dashed node, an unconfigured node is grey, and a stale node is yellow.

## Snapshots

`solve()` and `infer()` take the diagram directly, so you rarely need what happens under the hood: before running, they call `snapshot()`, which validates the diagram and returns a frozen view of its nodes, ordered so that parents come first. If the diagram has problems, `snapshot()` raises a `ValueError` that lists them, so you can't solve an incomplete diagram by accident:

```{code-cell} ipython3
diagram.snapshot().order
```

A snapshot still refers to the same node objects, so it is only valid until the diagram changes. Take a new one after editing.

```{admonition} Optional check: does a distribution use its parent?
:class: tip
`validate()` checks that a distribution's arguments match its parents, but it can't tell whether the function actually uses them. A common mistake, indexing a table with the wrong parent, passes this check. `probe_discrete_parents()` runs each distribution for every state of its discrete parents and warns, with a `DIST_IGNORES_PARENT` problem, when the result never changes. It is optional because it runs your code, and it only warns because a node can be independent of a parent on purpose. See [Chance nodes](chance_nodes.md) for details.
```
