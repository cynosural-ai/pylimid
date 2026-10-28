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

- **Continuous actions are scalar.** Decisions can choose finite categorical actions or a real number in a declared domain. [`DistributionPolicy`](https://pylimid.readthedocs.io/en/latest/api/policy.html#composing-continuous-policies) composes continuous distributions with parameter models. Vector actions and parameter-dependent support are not supported.
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

A small shop sells ice cream by weight. Each kilogram costs \$10 and brings in \$100 if it all sells, and anything not sold by the evening is thrown away. The freezer holds up to 40 kilograms. The owner can also run a promotion for \$60, which is expected to increase average demand by 20%.

The promotion must be announced the evening before, when tomorrow's weather is still uncertain. In the morning, the owner checks the weather and decides how much ice cream to order, remembering whether the promotion is running. Without a promotion, average demand is 4 kilograms on cloudy days and 12 on sunny days. Should the shop promote, and how much should it stock?

```python
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    DistributionPolicy,
    InfluenceDiagram,
    Interval,
    Linear,
    TabularSoftmax,
    UtilityNode,
    solve,
)

COST = 10.0
PRICE = 100.0
CAPACITY = 40.0
PROMOTION_COST = 60.0
PROMOTION_BOOST = jnp.array([1.0, 1.2])
WEATHER_STATES = ("cloudy", "sunny")
WEATHER_PROBS = jnp.array([0.7, 0.3])
DEMAND_MEANS = jnp.array([4.0, 12.0])


def demand_dist(weather, promotion):
    mean = DEMAND_MEANS[weather] * PROMOTION_BOOST[promotion]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


def profit(order, demand, promotion):
    revenue = PRICE * jnp.minimum(order, demand)
    return revenue - COST * order - PROMOTION_COST * promotion


diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=WEATHER_STATES,
        dist=lambda: dist.Categorical(probs=WEATHER_PROBS),
    )
)
diagram.add_node(
    DecisionNode(
        name="promotion",
        states=("skip", "run"),
        policy_family=TabularSoftmax(),
    )
)
diagram.add_node(
    ChanceNode(name="demand", parents=("weather", "promotion"), dist=demand_dist)
)
diagram.add_node(
    DecisionNode(
        name="order",
        parents=("weather", "promotion"),
        action_space=Interval(0.0, CAPACITY),
        policy_family=DistributionPolicy(
            distribution=dist.TruncatedNormal,
            parameter_model=Linear(outputs=("loc",)),
            fixed_parameters={"low": 0.0, "high": CAPACITY, "scale": 0.25},
        ),
    )
)
diagram.add_node(
    UtilityNode(name="profit", parents=("order", "demand", "promotion"), values=profit)
)

solution = solve(diagram, num_steps=2_000)
print(solution.render())
```

`weather` is a discrete chance node, and `demand` is a continuous one. The two decisions use different kinds of actions: `promotion` passes an index for `skip` or `run`, while `order` passes the quantity in kilograms directly to `profit`.

A decision's parents are what is known when it acts. The promotion has no parents because the weather is still unknown. The order sees both the weather and the earlier promotion choice, while actual demand remains unknown.

`TabularSoftmax` learns promotion probabilities, and `DistributionPolicy` learns a continuous order distribution. The interval declares which stock quantities are allowed, while the TruncatedNormal's fixed bounds keep orders within that range. `Linear` learns its location from the weather and promotion, with scale fixed at 0.25 kilograms.

In this run, the promotion rule puts more than 99.9% probability on running the promotion. When it runs, mean orders are about 7.2 kilograms on cloudy days and 21.0 on sunny days, with estimated expected profit of about \$542.6 per day. Both rules remain stochastic, and training does not guarantee the best policy.

The [quickstart](https://pylimid.readthedocs.io/en/latest/getting_started/quickstart.html) walks through this example and adds a Bayesian prior for a busier season. The [Solving guide](https://pylimid.readthedocs.io/en/latest/user_guide/solving.html) explores training settings and a different order distribution.

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
