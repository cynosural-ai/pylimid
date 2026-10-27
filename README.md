<p align="center">
  <img src="https://raw.githubusercontent.com/cynosural-ai/pylimid/main/docs/_static/branding/pylimid-readme.svg" alt="PyLIMID" width="560">
</p>

[![PyPI](https://img.shields.io/pypi/v/pylimid.svg)](https://pypi.org/project/pylimid/)
[![Documentation](https://readthedocs.org/projects/pylimid/badge/?version=latest)](https://pylimid.readthedocs.io/en/latest/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/cynosural-ai/pylimid/blob/main/LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

PyLIMID is a Python library built on top of [NumPyro](https://num.pyro.ai) for building and solving *limited-memory influence diagrams* (LIMIDs) whose chance nodes can be discrete, continuous, or both in the same model.

Rather than requiring every chance and utility node to be represented as a table, PyLIMID uses NumPyro distributions, [JAX](https://docs.jax.dev/en/latest/) functions, and sampling. And because JAX compiles and vectorises the computations, models run efficiently on both CPU and GPU.

> **Early development.** The public API is still settling and may change between releases, so pin a version if you build on it.

## Features

- **Flexible distributions.** A chance node is a NumPyro distribution whose parameters can be any JAX expression of its parents, so you are not limited to a conditional probability table.
- **Bayesian priors.** Represent uncertain model parameters as chance nodes with prior distributions.
- **Utilities as functions.** A utility is a Python function of its parents written with JAX, so it can express thresholds, nonlinearities, or risk preferences instead of a lookup table.
- **Policy fitting.** Policy gradient jointly fits the decision rules within declared policy families. Decisions can use categorical observations, numerical observations, or both, depending on the family.
- **Mixed inference.** Posteriors use exact enumeration for categorical nodes and supported finite counts, and MCMC sampling (NUTS) for continuous ones. Numerical counts also support forward simulation and observed likelihoods.

## Limitations

- **Decisions are discrete.** A decision node must choose among a finite set of actions. `TabularSoftmax` requires information parents with named finite states, while `LinearSoftmax` and custom families can use numerical observations.
- **Count posteriors need fixed enumerable support.** Scalar Binomial counts with a concrete number of trials are supported. Unknown unbounded counts, shifted ranges and batched counts remain unsupported. Parent-dependent bounds do not have a supported enumeration path.
- **Results are approximate.** Inference summaries and expected utility are estimated from samples. More evaluation samples reduce estimation noise, while policy quality also depends on the chosen family and training. Training does not guarantee an optimal policy. For discrete diagrams that meet its solver's requirements, consider [pyAgrum](https://pyagrum.readthedocs.io) when you need an exact solution.

## Installation

PyLIMID uses Python 3.12. Installing it also pulls in NumPyro and JAX:

```bash
pip install pylimid
```

The default installation uses the CPU. For NVIDIA GPUs on Linux (x86_64 or aarch64), choose a CUDA extra compatible with your GPU and NVIDIA driver:

```bash
pip install "pylimid[cuda12]"  # CUDA 12
pip install "pylimid[cuda13]"  # CUDA 13
```

These extras install the CUDA libraries automatically. They do not support native Windows or macOS. GPU support under Windows WSL2 is experimental. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for GPU and driver requirements.

## Example

A small shop sells ice cream. Every morning the owner checks the weather forecast and decides how many ice creams to stock, from 0 to 200 in steps of 20. Each ice cream costs \$1 and sells for \$10, and anything not sold by the evening is thrown away. The forecast is cloudy with probability 0.7 and sunny with probability 0.3. Past seasons suggest average demand of 40 ice creams on cloudy days and 120 on sunny days. The owner expects the coming season to be busier, but is uncertain about how much. How many should the shop stock?

We express that belief with a Bayesian prior on a positive `busyness` multiplier. Its LogNormal distribution has mean 1.2, representing 20% higher average demand, and a log-scale standard deviation of 0.1. Demand depends on both the weather and this unknown multiplier.

```python
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    TabularSoftmax,
    UtilityNode,
    solve,
)

WEATHER_STATES = ("cloudy", "sunny")
DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
ORDERS = jnp.arange(0, 201, 20)


def demand_dist(weather, busyness):
    mean = busyness * DEMAND_MEANS[weather]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked


diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=WEATHER_STATES,
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
diagram.add_node(
    ChanceNode(name="demand", parents=("weather", "busyness"), dist=demand_dist)
)
diagram.add_node(
    DecisionNode(
        name="order",
        parents=("weather",),
        states=tuple(str(n) for n in ORDERS),
        policy_family=TabularSoftmax(),
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

solution = solve(diagram)
print(solution.render())
```

`weather` is a discrete chance node, `busyness` and `demand` are continuous ones, `order` is the decision, and `profit` is the utility. The decision's parents are what is known when it is made, so `order` sees the weather while `busyness` and `demand` remain unknown. Note that `order` is the index of the chosen action, which is why `profit` looks the stock level up in `ORDERS`.

`TabularSoftmax` learns a separate set of action probabilities for each weather forecast. The fitted rule puts most probability on stocking 80 ice creams on cloudy days and 200 on sunny days. Following it means sampling an order from those probabilities, and the displayed expected profit estimates the result of following the whole rule.

These stock levels are far above average demand: a missed sale loses \$9 of margin, while a leftover ice cream only costs \$1. The most likely sunny-day order reaches the menu's maximum of 200. The [quickstart](https://pylimid.readthedocs.io/en/latest/getting_started/quickstart.html) walks through this example step by step, and the [Solving guide](https://pylimid.readthedocs.io/en/latest/user_guide/solving.html) explores a wider menu and training settings.

## Documentation

The full documentation is at **<https://pylimid.readthedocs.io>**:

- The [quickstart](https://pylimid.readthedocs.io/en/latest/getting_started/quickstart.html) builds and solves the example above.
- The [user guide](https://pylimid.readthedocs.io/en/latest/user_guide/building_a_diagram.html) explains each piece in depth: chance nodes, decisions, utilities, inference, and solving.

<!--
- The [tutorials](https://pylimid.readthedocs.io/en/latest/tutorials/oil_field.html) build complete models step by step.
-->

## Contributing

Contributions are welcome. [CONTRIBUTING.md](https://github.com/cynosural-ai/pylimid/blob/main/CONTRIBUTING.md) explains how to set up a development environment, which checks a change has to pass, and the conventions the project follows.

## License

PyLIMID is released under the Apache 2.0 license. See [LICENSE.md](https://github.com/cynosural-ai/pylimid/blob/main/LICENSE.md) for details.
