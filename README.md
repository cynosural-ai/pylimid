# pylimid

**Mixed-type limited-memory influence diagrams, on NumPyro.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

pylimid is a Python library for decision models that mix discrete and continuous variables. You describe chance, decision, and utility nodes; it validates the diagram, infers posteriors, and returns the optimal policy.

Exact tabular solvers like pyAgrum are the right tool for fully discrete influence diagrams, but they cannot express a continuous variable or a payoff that is not a table. pylimid takes the other trade. Distributions, decisions, and utilities are ordinary NumPyro callables evaluated with JAX, and every answer is a Monte-Carlo estimate from a solver with a stated guarantee. That makes reachable models that tabular solvers cannot express: arbitrary continuous distributions, mixed diagrams, and payoffs like `jnp.log(wealth)`.

> **Early development.** pylimid is pre-1.0; the public API may change between releases. Pin a version if you build on it.

## Capabilities

- **Mixed-type by construction.** Chance nodes can be discrete, continuous, or mixed, using any NumPyro distribution.
- **Callables, not tables.** A distribution or utility is a Python function of its parents, so a payoff can be any JAX-traceable expression.
- **Explicit information sets.** A decision observes exactly the parents you draw — there is no implicit no-forgetting. Anything an agent should remember is a memory arc you add, and a classical influence diagram is the special case where all of them are drawn.
- **Two verbs.** `infer` computes posterior draws — exact enumeration for discrete sites, NUTS for continuous ones — and `solve` returns an optimal policy with its expected utility.
- **Monte-Carlo with stated guarantees.** Backward induction on solvable diagrams, an exhaustive scan otherwise; the result records which solver ran.
- **A mutable workspace.** Build incrementally, wire and rewire edges, and gate inference behind an explicit `validate()` / `snapshot()` checkpoint.

## Decision problem example

A newsvendor stocks newspapers before the day's demand is known. Each paper costs \$1 and sells for \$10; leftovers are worthless. Demand is a right-skewed Gamma — quiet days average 40, busy days average 120 — and the profit is an arbitrary function of what is stocked and demanded.

```python
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

orders = jnp.arange(0, 261, 20)  # discrete stock levels
rates = jnp.array([0.1, 1 / 30])  # Gamma rate: quiet mean 40, busy mean 120

diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="day",
        states=("quiet", "busy"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("day",),
        dist=lambda day: dist.Gamma(concentration=4.0, rate=rates[day]),
    )
)
diagram.add_node(
    DecisionNode(name="order", parents=("day",), states=tuple(str(n) for n in orders))
)
diagram.add_node(
    UtilityNode(
        name="profit",
        parents=("order", "demand"),
        values=lambda order, demand: (
            10.0 * jnp.minimum(orders[order], demand) - 1.0 * orders[order]
        ),
    )
)

solution = solve(diagram)
solution.policy  # stock 60 on a quiet day, 200 on a busy day
```

The solver stocks well above the average demand — 60 on a quiet day (mean 40), 200 on a busy day (mean 120) — because a missed sale forfeits the \$9 margin while an unsold paper costs only \$1, so the optimum tracks the 90th percentile. The [newsvendor tutorial](docs/tutorials/newsvendor.md) walks through the full derivation.

## Installation

Requires Python 3.12. pylimid is not on PyPI yet:

```bash
pip install "pylimid @ git+https://github.com/cynosural-ai/pylimid.git"
```

For a development checkout, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

- [Quickstart](docs/getting_started/quickstart.md)
- [User guide](docs/user_guide/building_a_diagram.md)
- [Tutorials](docs/tutorials/oil_field.md)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the checks, and the conventions.

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
