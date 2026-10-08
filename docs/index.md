# PyLIMID

PyLIMID is a Python library built on top of [NumPyro](https://num.pyro.ai) for building and solving *limited-memory influence diagrams* (LIMIDs) whose chance nodes can be discrete, continuous, or both in the same model.

Rather than requiring every chance and utility nodes to be represented as a table, PyLIMID uses [NumPyro](https://num.pyro.ai) distributions, [JAX](https://docs.jax.dev/en/latest/) functions, and sampling. Also, by compiling and vectorising computations with [JAX](https://docs.jax.dev/en/latest/), models can run efficiently on both CPU and GPU.

## Features

- **Flexible distributions.** A chance node is a NumPyro distribution whose parameters can be any JAX expression of its parents, so you are not limited to a conditional probability table.
- **Utilities as functions.** A utility is a Python function of its parents written with JAX, so it can express thresholds, nonlinearities, or risk preferences instead of a lookup table.
- **Mixed inference.** Posteriors are computed by exact enumeration for discrete nodes and by MCMC sampling (NUTS) for continuous ones, so a single diagram can combine both.

## Limitations

- **Decisions are discrete.** A decision node must choose among a finite set of actions (continuous decisions are not *yet* supported).
- **Results are approximate.** Inference and solving both return Monte-Carlo estimates rather than exact answers, and they get more precise with more samples. For purely discrete diagrams, an exact solver like [pyAgrum](https://pyagrum.readthedocs.io) is the better choice.

```{admonition} Early development
:class: warning
The public API is still settling and may change between releases.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:text-align: center
Start here: build, validate, and solve a small influence diagram.

+++
[](getting_started/quickstart.md)
:::

:::{grid-item-card} User guide
:text-align: center
How each piece works: diagrams, chance nodes, decisions, utilities, inference, and solving.

+++
[](user_guide/building_a_diagram.md)
:::

:::{grid-item-card} Tutorials
:text-align: center
Build complete models step by step, starting with a classic decision problem.

+++
[](tutorials/oil_field.md)
:::

:::{grid-item-card} API reference
:text-align: center
Technical descriptions of every public class, function, and method.

+++
[](api/index)
:::

::::

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 1

getting_started/installation
getting_started/quickstart
```

```{toctree}
:hidden:
:caption: User guide
:maxdepth: 1

user_guide/building_a_diagram
user_guide/chance_nodes
user_guide/decisions_and_information_sets
user_guide/utilities
user_guide/inference
user_guide/solving
user_guide/scope_and_limitations
```

```{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 1

tutorials/oil_field
tutorials/logistics_center
tutorials/newsvendor
tutorials/bayesian_networks
```

```{toctree}
:hidden:
:caption: API reference
:maxdepth: 1

api/index
api/pylimid
api/graph
api/solvers
```
