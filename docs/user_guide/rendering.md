---
file_format: mystnb
kernelspec:
  name: python3
---

# Rendering

A diagram renders itself in two formats, from the same underlying picture: **Mermaid** text for reading and for LLMs, and **SVG** for notebooks and documents. Chance nodes are circles, decision nodes rectangles, and utility nodes diamonds.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram

weather = InfluenceDiagram()
weather.add_node(
    ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
    )
)
weather.add_node(
    ChanceNode(
        name="wet_grass",
        parents=("rain",),
        states=("dry", "wet"),
        dist=lambda rain: dist.Categorical(
            probs=jnp.array([[0.9, 0.1], [0.2, 0.8]])[rain]
        ),
    )
)
weather.validate()
```

## Mermaid: text-first

`to_mermaid()` returns a deterministic Mermaid `flowchart` source string — no runtime dependency, stable output, and a format that GitHub, chat UIs, and LLMs all handle. Nodes are emitted in topological order and edges follow each child's declared parent order, so the same diagram always produces the same source.

```{code-cell} ipython3
print(weather.to_mermaid())
```

A consistent diagram is exactly the nodes and the edges: no styling, no decoration.

## SVG: for notebooks and documents

`to_svg()` renders through the Graphviz `dot` binary, which must be on `PATH`. A diagram in Jupyter renders itself: the last expression in a cell triggers `_repr_svg_`, which calls `to_svg()`. Without `dot`, `_repr_svg_` returns `None` and Jupyter falls back to the plain repr; an explicit `to_svg()` call fails loudly instead.

```{code-cell} ipython3
weather
```

## Rendering an unfinished workspace

Rendering never validates. It draws the **live workspace**, so a model caught mid-edit is still visible: a dangling parent becomes a dashed ghost node and a node that is not `CONSISTENT` is tinted. Add a puddle that depends on rain and on a drainage node that does not exist yet:

```{code-cell} ipython3
weather.add_node(
    ChanceNode(name="puddle", parents=("rain", "drainage"), states=("small", "large"))
)
print(weather.to_mermaid())
```

`puddle` is `UNCONFIGURED` (no `dist` yet), so it is grey; `drainage` is not in the diagram, so it appears as a dashed ghost with a dashed edge to `puddle`. The styles are only emitted for the classes actually used, and the figure shows the same picture:

```{code-cell} ipython3
weather
```

## What each format is for

- **Mermaid** — text: diffs, logs, LLM prompts, GitHub rendering.
- **SVG** — figures: notebooks, documentation pages, anything that should show the picture without JavaScript.

They share one backend-neutral view of the graph, so the two never disagree on shapes, ordering, dangling nodes, or consistency tints.
