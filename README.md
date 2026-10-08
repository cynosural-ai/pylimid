# PyLIMID

[![PyPI](https://img.shields.io/pypi/v/pylimid.svg)](https://pypi.org/project/pylimid/)
[![Documentation](https://readthedocs.org/projects/pylimid/badge/?version=latest)](https://pylimid.readthedocs.io/en/latest/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

PyLIMID is a Python library built on top of [NumPyro](https://num.pyro.ai) for building and solving *limited-memory influence diagrams* (LIMIDs) whose chance nodes can be discrete, continuous, or both in the same model.

Rather than requiring every chance and utility node to be represented as a table, PyLIMID uses NumPyro distributions, [JAX](https://docs.jax.dev/en/latest/) functions, and sampling. And because JAX compiles and vectorises the computations, models run efficiently on both CPU and GPU.

> **Early development.** The public API is still settling and may change between releases, so pin a version if you build on it.

## Features

- **Flexible distributions.** A chance node is a NumPyro distribution whose parameters can be any JAX expression of its parents, so you are not limited to a conditional probability table.
- **Utilities as functions.** A utility is a Python function of its parents written with JAX, so it can express thresholds, nonlinearities, or risk preferences instead of a lookup table.
- **Mixed inference.** Posteriors are computed by exact enumeration for discrete nodes and by MCMC sampling (NUTS) for continuous ones, so a single diagram can combine both.

## Limitations

- **Decisions are discrete.** A decision node must choose among a finite set of actions (continuous decisions are not *yet* supported).
- **Results are approximate.** Inference and solving both return Monte Carlo estimates rather than exact answers, and they get more precise with more samples. For purely discrete diagrams, an exact solver like [pyAgrum](https://pyagrum.readthedocs.io) is the better choice.

## Installation

PyLIMID uses Python 3.12. Installing it also pulls in NumPyro and JAX:

```bash
pip install pylimid
```

By default, JAX is installed in its CPU-only version. If you have an NVIDIA GPU, install the extra that matches your CUDA version:

```bash
pip install "pylimid[cuda12]"  # CUDA 12
pip install "pylimid[cuda13]"  # CUDA 13, for newer drivers
```

## Example

A street vendor sells ice cream. Every morning they check the weather forecast and decide how many ice creams to stock, from 0 to 200 in steps of 20. Each ice cream costs \$1 and sells for \$10, and anything not sold by the evening is thrown away. The forecast is cloudy with probability 0.7 and sunny with probability 0.3, and demand is uncertain either way: about 40 ice creams on an average cloudy day and 120 on an average sunny one. How many should the vendor stock?

```python
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

WEATHER_STATES = ("cloudy", "sunny")
DEMAND_MEANS = jnp.array([40.0, 120.0])
ORDERS = jnp.arange(0, 201, 20)


def demand_dist(weather):
    return dist.Gamma(concentration=4.0, rate=4.0 / DEMAND_MEANS[weather])


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
diagram.add_node(ChanceNode(name="demand", parents=("weather",), dist=demand_dist))
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

solution = solve(diagram, num_samples=50_000)
for info, action in solution.policy["order"].items():
    print(f"{WEATHER_STATES[info[0]]}: stock {int(ORDERS[action])}")
# cloudy: stock 60
# sunny: stock 200
```

`weather` is a discrete chance node, `demand` is a continuous one, `order` is the decision, and `profit` is the utility. The decision's parents are what is known when it is made, so `order` sees the weather but not the demand. Note that `order` is the index of the chosen action, which is why `profit` looks the stock level up in `ORDERS`.

The vendor should stock far more than the average demand: a missed sale loses \$9 of margin, while a leftover ice cream only costs \$1. The [quickstart](https://pylimid.readthedocs.io/en/latest/getting_started/quickstart.html) walks through this example step by step.

## Documentation

The full documentation is at **<https://pylimid.readthedocs.io>**:

- The [quickstart](https://pylimid.readthedocs.io/en/latest/getting_started/quickstart.html) builds and solves the example above.
- The [user guide](https://pylimid.readthedocs.io/en/latest/user_guide/building_a_diagram.html) explains each piece in depth: chance nodes, decisions, utilities, inference, and solving.
- The [tutorials](https://pylimid.readthedocs.io/en/latest/tutorials/oil_field.html) build complete models step by step.

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up a development environment, which checks a change has to pass, and the conventions the project follows.

## License

PyLIMID is released under the Apache 2.0 license. See [LICENSE.md](LICENSE.md) for details.
