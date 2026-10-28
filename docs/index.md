# PyLIMID

PyLIMID is a Python library built on top of [NumPyro](https://num.pyro.ai) for building and solving *limited-memory influence diagrams* (LIMIDs) whose chance nodes can be discrete, continuous, or both in the same model.

Rather than requiring every chance and utility node to be represented as a table, PyLIMID uses [NumPyro](https://num.pyro.ai) distributions, [JAX](https://docs.jax.dev/en/latest/) functions, and sampling. Also, by compiling and vectorising computations with [JAX](https://docs.jax.dev/en/latest/), models can run efficiently on both CPU and GPU.

## Features

- **Flexible distributions.** A chance node is a NumPyro distribution whose parameters can be any JAX expression of its parents, so you are not limited to a conditional probability table.
- **Bayesian priors.** Represent uncertain model parameters as chance nodes with prior distributions.
- **Utilities as functions.** A utility is a Python function of its parents written with JAX, so it can express thresholds, nonlinearities, or risk preferences instead of a lookup table.
- **Policy fitting.** {py:func}`solve` uses policy gradient to jointly fit the decision rules within declared policy families. Decisions can use categorical observations, numerical observations, or both, depending on the family.
- **Mixed inference.** Posteriors use exact enumeration for categorical nodes and supported finite counts, and MCMC sampling (NUTS) for continuous ones. Numerical counts support forward simulation and observed likelihoods, but unbounded count latents remain unsupported.

## Limitations

- **Continuous actions are scalar.** Choose finite categorical actions through `states`, or declare a {py:class}`Real` or {py:class}`Interval` action space with a compatible continuous policy family. Vector actions and parameter-dependent support are not supported.
- **Results are approximate.** Inference summaries and expected utility are estimated from samples. More evaluation samples reduce estimation noise, while policy quality also depends on the chosen family and training. Training does not guarantee an optimal policy. The [Solving guide](user_guide/solving.md#how-much-can-we-trust-the-result) explains how to assess fitted policies. For discrete diagrams that meet its solver's requirements, consider [pyAgrum](https://pyagrum.readthedocs.io) when you need an exact solution.

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

:::{grid-item-card} API reference
:text-align: center
Technical descriptions of every public class, function, and method.

+++
[](api/index)
:::

::::

<!--
:::{grid-item-card} Tutorials
:text-align: center
Build complete models step by step, starting with a classic decision problem.

+++
[](tutorials/oil_field.md)
:::
-->

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
user_guide/distributions_and_utilities
user_guide/solving
user_guide/inference
```

<!--
```{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 1

tutorials/oil_field
tutorials/logistics_center
tutorials/newsvendor
tutorials/bayesian_networks
```
-->

```{toctree}
:hidden:
:caption: Resources
:maxdepth: 1

resources/common_pitfalls
resources/count_distributions
```

```{toctree}
:hidden:
:caption: API reference
:maxdepth: 1

api/index
api/pylimid
api/graph
api/policy
```
