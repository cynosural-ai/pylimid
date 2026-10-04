# pylimid

**pylimid** models and solves *limited-memory influence diagrams* (LIMIDs) whose chance variables can be discrete, continuous, or mixed.

A decision's information set is exactly the parents you draw — there is no implicit no-forgetting — and any past information an agent needs is an explicit memory arc. Chance distributions, decisions, and utilities are ordinary Python callables, so a model can mix types freely and a payoff can be any function of its parents. A classical influence diagram is the special case where all the memory arcs happen to be drawn.

The library is built on [NumPyro](https://num.pyro.ai). Distributions and utilities are traced and evaluated with JAX, posterior inference enumerates discrete sites or runs NUTS, and the optimal decisions are found by Monte-Carlo solvers with a stated guarantee per solver. That is what lets `pylimid` reach models exact tabular solvers cannot: arbitrary continuous distributions, mixed diagrams, and utilities that are not tables.

```{admonition} Early development
:class: warning
pylimid is pre-1.0. The public API is still settling and may change between releases; pin a version if you build on it.
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
user_guide/rendering
user_guide/scope_and_limitations
```

```{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 1

tutorials/oil_field
tutorials/logistics_center
tutorials/newsvendor
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
