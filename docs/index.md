---
file_format: mystnb
kernelspec:
  name: python3
---

# pylimid

**pylimid** models and solves *limited-memory influence diagrams* (LIMIDs) whose chance variables can be discrete, continuous, or mixed. Chance distributions, decisions and utilities are ordinary Python callables backed by [NumPyro](https://num.pyro.ai), and the optimal decisions are found by Monte-Carlo solvers.

```{admonition} Early development
:class: warning
pylimid is pre-1.0. The public API is still settling and may change between releases; pin a version if you build on it.
```

## Install

pylimid is not on PyPI yet. Install it from source:

```bash
pip install "pylimid @ git+https://github.com/cynosural-ai/decisionpy.git"
```

See [](getting_started/installation.md) for the development setup and the JAX CPU/GPU notes.

## A first diagram

An influence diagram is a graph of chance, decision and utility nodes. Here a patient may be sick; the doctor decides whether to treat; the patient may recover. Recovering is worth 100, treating costs 20.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist
from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

diagram = InfluenceDiagram()
diagram.add_node(ChanceNode(
    name="disease", states=("healthy", "sick"),
    dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
))
diagram.add_node(DecisionNode(name="treat", parents=("disease",), states=("no", "yes")))
diagram.add_node(ChanceNode(
    name="recovery", parents=("disease", "treat"), states=("no", "yes"),
    dist=lambda disease, treat: dist.Categorical(probs=jnp.array([
        [[0.10, 0.90], [0.08, 0.92]],
        [[0.60, 0.40], [0.20, 0.80]],
    ])[disease, treat]),
))
diagram.add_node(UtilityNode(
    name="utility", parents=("recovery", "treat"),
    values=lambda recovery, treat: 100.0 * recovery - 20.0 * treat,
))

solution = solve(diagram)
print(solution.policy, round(solution.expected_utility, 2))
```

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:text-align: center
Build a diagram, validate it, and solve it in five minutes.

+++
[](getting_started/quickstart.md)
:::

:::{grid-item-card} API reference
:text-align: center
Signatures and docstrings for every public name.

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
:caption: Tutorials
:maxdepth: 1

tutorials/oil_field
```

```{toctree}
:hidden:
:caption: Examples
:maxdepth: 1

examples/influence_diagrams/medical_treatment
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
