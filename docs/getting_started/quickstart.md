---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

This quickstart shows how to use PyLIMID to model and solve a small decision problem.

## The problem

A small shop sells ice cream. Every morning, before opening, the owner checks the weather forecast and decides how many packages of ice cream to stock for the day. Each package contains 20 ice creams, and each ice cream costs \$1 and sells for \$10. Now, since ice cream doesn't last very long, anything not sold by the evening is thrown away.

```{figure} /_static/figures/quickstart/ice_cream_shop.jpg
:alt: A street vendor selling ice cream
:width: 70%
```

The forecast for today is cloudy with probability 0.7 and sunny with probability 0.3. Demand is uncertain either way, but it is higher when it is sunny: about 40 ice creams on an average cloudy day and 120 on an average sunny one. **How many ice creams should the shop stock?**

```{admonition} Distributions origin
:class: info

The probabilities and demand distributions below represent the owner's estimates, informed by observations from previous seasons. We use these estimates directly to build the model.
```

From this definition, we can discern 4 main variables in this problem:

- `weather`: a discrete chance variable with two values cloudy or sunny.
- `demand`: a continuous chance variable whose distribution depends on the weather.
- `order`: a decision variable of how many ice creams to stock, which we choose knowing the weather but not the demand. 
- `profit`: a utility variable, which represents what the shop earns, given the order and the demand.

## Building the diagram

We start with the numbers. The owner chooses from a menu of stock levels, from 0 to 200 in steps of 20:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

COST = 1.0
PRICE = 10.0
WEATHER_STATES = ("cloudy", "sunny")
WEATHER_PROBS = jnp.array([0.7, 0.3])
DEMAND_MEANS = jnp.array([40.0, 120.0])
ORDERS = jnp.arange(0, 201, 20)
```

Demand is always positive, and most days are quiet while a few are very busy, so we model it with a {py:class}`Gamma distribution <numpyro.distributions.continuous.Gamma>` whose mean depends on the weather:

```{code-cell} ipython3
def demand_dist(weather):
    return dist.Gamma(concentration=4.0, rate=4.0 / DEMAND_MEANS[weather])
```

Plotting the two distributions shows what the shop is up against. Sunny days are not only busier on average, they are also much less predictable, and on both kinds of day there is a long tail of unusually busy days to the right of the mean (dotted line):

```{figure} /_static/figures/quickstart/demand.png
:alt: Demand distributions for cloudy and sunny days, with dotted lines at their means
:width: 80%

Demand on cloudy and sunny days. The dotted lines mark the average demand.
```

The profit is a plain function: \$10 for every ice cream sold, minus \$1 for every ice cream stocked.

```{code-cell} ipython3
def profit(order, demand):
    stocked = ORDERS[order]
    return PRICE * jnp.minimum(stocked, demand) - COST * stocked
```

Note that `order` is the *index* of the chosen action, not the stock level itself, which is why `profit` looks it up in `ORDERS`. The same goes for `weather`: `0` is cloudy and `1` is sunny.

Now we put the four nodes into an {py:class}`InfluenceDiagram`: `weather` and `demand` are each a {py:class}`ChanceNode`, `order` is a {py:class}`DecisionNode`, and `profit` is a {py:class}`UtilityNode`. A node's `parents` are the nodes it depends on. For a decision, the parents are what is known when the decision is made, so `order` lists `weather` but not `demand`:

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=WEATHER_STATES,
        dist=lambda: dist.Categorical(probs=WEATHER_PROBS),
    )
)
diagram.add_node(ChanceNode(name="demand", parents=("weather",), dist=demand_dist))
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
```

## Validating the model

{py:meth}`~InfluenceDiagram.validate` checks the diagram's structure and node configuration, reporting problems such as missing parents, cycles or unconfigured nodes. It does not evaluate the distributions or utility functions though. An empty list means these checks passed:

```{code-cell} ipython3
diagram.validate()
```

A valid diagram can also be drawn. Chance nodes are circles, decisions are rectangles, and utilities are diamonds:

```{code-cell} ipython3
diagram
```

## Solving the model

{py:func}`solve` finds the policy with the highest expected profit, that is, how many ice creams to stock for each forecast. It returns a {py:class}`Solution`, whose `policy` maps each forecast to the index of the best order and whose `expected_utility` is the expected profit under that policy:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)

for info, action in solution.policy["order"].items():
    print(f"{WEATHER_STATES[info[0]]}: stock {int(ORDERS[action])}")
print(f"expected profit: {float(solution.expected_utility):.1f}")
```

The shop should stock 60 ice creams on a cloudy day and 200 on a sunny day. This is far more than the average demand. Given the high margin on ice cream sales, it pays to carry extra stock in case the day is busy.

Both the policy and the expected profit are Monte Carlo estimates, computed by simulating `num_samples` days. More samples give more precise results, and a run is reproducible: the same model and settings always give the same answer.

## Adding a Bayesian prior

The owner's past experience suggests averages of 40 ice creams on cloudy days and 120 on sunny days. The owner expects the coming season to be busier, but is uncertain about how much. A **Bayesian prior** can be used to express that belief.

We represent it with an unknown `busyness` multiplier, which can take any positive value. We give it a {py:class}`LogNormal distribution <numpyro.distributions.continuous.LogNormal>` with mean 1.2, representing 20% higher average demand, i.e., 48 ice creams on cloudy days and 144 on sunny days. A log-scale standard deviation of 0.1 puts about 95% of its probability within roughly 20% of the new mean.

```{figure} /_static/figures/quickstart/busyness_prior.png
:alt: LogNormal density for the busyness multiplier, with mean 1.2 marked by a dotted line and about 95 percent of its probability shaded between 0.96 and 1.44
:width: 80%

The prior for `busyness`, with about 95% of its probability in the shaded range. Values below 1 describe a quieter season; values above 1 describe a busier one.
```

We copy the diagram, add `busyness` as a parent of `demand`, and update the demand distribution:

```{code-cell} ipython3
from copy import deepcopy

BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1

prior_diagram = deepcopy(diagram)
prior_diagram.add_node(
    ChanceNode(
        name="busyness",
        dist=lambda: dist.LogNormal(
            jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
        ),
    )
)
prior_diagram.add_arc("busyness", "demand")


def uncertain_demand_dist(weather, busyness):
    mean = busyness * DEMAND_MEANS[weather]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


prior_diagram.set_dist("demand", uncertain_demand_dist)
prior_diagram
```

{py:func}`solve` averages over possible `busyness` values when comparing stock levels:

```{code-cell} ipython3
prior_solution = solve(prior_diagram, num_samples=50_000)

for info, action in prior_solution.policy["order"].items():
    print(f"{WEATHER_STATES[info[0]]}: stock {int(ORDERS[action])}")
print(f"expected profit: {prior_solution.expected_utility:.1f}")
```

With this belief, the recommended cloudy-day stock rises from 60 to 80 ice creams. The sunny-day order stays at the menu's maximum of 200.

## Next steps

- The [user guide](../user_guide/building_a_diagram.md) explains how to build a diagram and how to solve it.
