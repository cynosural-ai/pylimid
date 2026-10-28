---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

This quickstart shows how to use PyLIMID to model and solve a small decision problem.

## The problem

A small shop sells ice cream. Every morning, before opening, the owner checks the weather forecast and decides how much ice cream to order from a supplier. Each kilogram costs \$10 and brings in \$100 if it all sells. Now, since ice cream doesn't last very long, anything not sold by the evening is thrown away. The shop's freezer holds up to 40 kilograms, and the supplier accepts orders by weight, so the owner can order, for example, 8.3 kilograms.

```{figure} /_static/figures/quickstart/ice_cream_shop.jpg
:alt: A small ice cream shop
:width: 70%
```

The owner is also considering a daily promotion that costs \$60 and is expected to increase average demand by 20%. However, it must be announced the evening before, when the next day's weather is still uncertain. Thus, there are two decisions: whether to run the promotion, and then how much ice cream to order once the weather is known.

The forecast for tomorrow is cloudy with probability 0.7 and sunny with probability 0.3. Demand is uncertain either way, but it is higher when it is sunny: without a promotion, about 4 kilograms on an average cloudy day and 12 on an average sunny one. **Should the shop run the promotion, and how much ice cream should it stock?**

```{admonition} Where the estimates come from
:class: info

The weather probabilities, demand distributions and promotion effect represent the owner's estimates, informed by previous seasons. We use these estimates directly to build the model.
```

From this definition, we can discern five main variables in the problem:

- `weather`: a discrete chance variable, either `cloudy` or `sunny`.
- `promotion`: a categorical decision of whether to `skip` or `run` the promotion, made before the weather is known.
- `demand`: a continuous chance variable measured in kilograms, whose distribution depends on the weather and the promotion.
- `order`: a continuous decision measured in kilograms, made knowing the weather and the earlier promotion choice, but not the demand.
- `profit`: a utility variable representing sales revenue minus the stock and promotion costs.

## Building the diagram

We start with the prices, the freezer's capacity and the estimates for weather and demand:

```{code-cell} ipython3
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
```

Demand is always positive, and most days are quiet while a few are very busy, so we model it with a {py:class}`Gamma distribution <numpyro.distributions.continuous.Gamma>` whose mean depends on the weather and the promotion:

```{code-cell} ipython3
def demand_dist(weather, promotion):
    mean = DEMAND_MEANS[weather] * PROMOTION_BOOST[promotion]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)
```

Note that the categorical variables pass indices: `weather=0` means cloudy and `promotion=1` means the promotion runs. We use those indices to look up the demand mean and promotion effect.

Profit is the revenue from whatever sells, minus the cost of everything ordered and the promotion if it runs:

```{code-cell} ipython3
def profit(order, demand, promotion):
    revenue = PRICE * jnp.minimum(order, demand)
    return revenue - COST * order - PROMOTION_COST * promotion
```

Here, `order` and `demand` are quantities in kilograms. The continuous decision passes the amount ordered directly to `profit`.

Now we put the five nodes into an {py:class}`InfluenceDiagram`: `weather` and `demand` are each a {py:class}`ChanceNode`, `promotion` and `order` are each a {py:class}`DecisionNode`, and `profit` is a {py:class}`UtilityNode`. A node's `parents` are the nodes it depends on. For a decision, the parents are what is known when the choice is made. Thus, `promotion` has no parents, while `order` lists both `weather` and `promotion`.

We also give each decision a policy family, which describes what we want to fit. For `promotion`, {py:class}`TabularSoftmax` lets us learn probabilities for the two actions, `skip` and `run`.

For `order`, we use {py:class}`DistributionPolicy` to learn a distribution over the amount ordered. We choose a {py:class}`TruncatedNormal <numpyro.distributions.truncated.TruncatedNormal>` to keep orders between zero and the freezer's capacity. {py:class}`Linear` learns how its location depends on the weather and promotion, while we hold its scale at 0.25 kilograms:

```{code-cell} ipython3
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
    UtilityNode(
        name="profit", parents=("order", "demand", "promotion"), values=profit
    )
)
```

The {py:class}`Interval` declares the allowed stock quantities, and the truncation bounds keep the policy's orders within that range. Fixing the scale leaves a small spread in the orders while the solver learns their location. Here, location and scale refer to the underlying Normal before truncation.

## Validating the model

{py:meth}`~InfluenceDiagram.validate` checks the diagram's structure and node configuration, including missing parents, cycles and unconfigured nodes. It does not evaluate the distributions or utility function. An empty list means these checks passed:

```{code-cell} ipython3
diagram.validate()
```

A valid diagram can also be drawn. Chance nodes are circles, decisions are rectangles, and utilities are diamonds:

```{code-cell} ipython3
diagram
```

Notice that `promotion` is a parent of both `demand` and `order`. For `demand`, the parent changes the distribution. For `order`, it says that the owner remembers the earlier choice. The promotion cannot use tomorrow's weather, and neither decision knows the demand when it acts.

## Solving the model

{py:func}`solve` fits both policies together to increase expected profit. It returns a {py:class}`Solution`, whose `policy` contains a fitted rule for each decision and whose `expected_utility` estimates profit when the shop follows both rules. We allow 2,000 training updates for this example:

```{code-cell} ipython3
solution = solve(diagram, num_steps=2_000)
print(solution.render())
```

The promotion rule is a fitted {py:class}`CategoricalPolicy`. We can read its probabilities in the order of the decision's states, `skip` and `run`:

```{code-cell} ipython3
promotion_rule = solution.policy["promotion"]
print("Promotion probabilities [skip, run]:", promotion_rule.probabilities())
```

The order rule is a fitted {py:class}`ContinuousPolicy`. Its `distribution` method returns the order distribution for a particular situation. For example, we can read the mean order on cloudy and sunny days when the promotion runs:

```{code-cell} ipython3
order_rule = solution.policy["order"]
for weather, label in enumerate(WEATHER_STATES):
    orders = order_rule.distribution(
        weather=jnp.array(weather), promotion=jnp.array(1)
    )
    print(f"{label}, promotion running: mean order {orders.mean:.1f} kg")
```

In this run, the promotion rule puts more than 99.9% probability on running the promotion. When it runs, the mean orders are about 7.2 kilograms on cloudy days and 21.0 on sunny days. This is far more than the average demand. Given the high margin on ice cream sales, it pays to carry extra stock in case the day is busy. The estimated expected profit is about \$542.6 per day.

We can also plot the promotion probabilities and the two order distributions:

```{figure} /_static/figures/quickstart/order_policy.png
:alt: The probability of skipping or running the promotion, followed by cloudy-day and sunny-day order densities when the promotion runs
:width: 100%

The fitted promotion probabilities and continuous order distributions when the promotion runs.
```

Both rules remain stochastic: following them means sampling whether to run the promotion, then sampling an order given the weather and the earlier choice. The expected profit is estimated under both rules, rather than always running the promotion and ordering the displayed means.

By default, each training update uses 10,000 simulated days. The default random key makes repeated calls with the same model and settings reproducible.

However, training does not guarantee the *best* policy. More samples reduce noise in the profit estimate, but cannot fix incomplete training. Situations that rarely appear during simulation, such as skipping the promotion in this run, may also receive little training.

## Adding a Bayesian prior

The owner's past experience suggests averages of 4 kilograms on cloudy days and 12 on sunny days, before any promotion. The owner expects the coming season to be busier, but is uncertain about how much. A **Bayesian prior** can be used to express that belief.

We introduce an unknown `busyness` multiplier with a {py:class}`LogNormal distribution <numpyro.distributions.continuous.LogNormal>`. Its mean is 1.2, representing 20% higher average demand, with a log-scale standard deviation of 0.1:

```{figure} /_static/figures/quickstart/busyness_prior.png
:alt: LogNormal density for the busyness multiplier, with mean 1.2 marked and about 95 percent of its probability shaded
:width: 80%

The prior for seasonal busyness. The promotion's effect is an additional multiplier.
```

We add `busyness` to the diagram as a parent of `demand`, and update the demand distribution:

```{code-cell} ipython3
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1

diagram.add_node(
    ChanceNode(
        name="busyness",
        dist=lambda: dist.LogNormal(
            jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
        ),
    )
)
diagram.add_arc("busyness", "demand")


def uncertain_demand_dist(weather, promotion, busyness):
    mean = DEMAND_MEANS[weather] * PROMOTION_BOOST[promotion] * busyness
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


diagram.set_dist("demand", uncertain_demand_dist)
diagram
```

{py:func}`solve` now samples possible `busyness` values while fitting the two policies and estimating their expected profit. We keep the same policy families because the owner still has the same information at each choice: `busyness` is unknown to both decisions, and the weather is only known when ordering:

```{code-cell} ipython3
prior_solution = solve(diagram, num_steps=2_000)
print(prior_solution.render())

for weather, label in enumerate(WEATHER_STATES):
    orders = prior_solution.policy["order"].distribution(
        weather=jnp.array(weather), promotion=jnp.array(1)
    )
    print(f"{label}, promotion running: mean order {orders.mean:.1f} kg")
```

With this belief, the fitted rule still almost always runs the promotion. The mean orders when it runs rise to about 8.5 kilograms on cloudy days and 27.0 on sunny days. The estimated expected profit is about \$660.9 per day under this busier-season model.

```{figure} /_static/figures/quickstart/prior_order_policy.png
:alt: Fitted promotion probabilities and weather-dependent continuous order densities with uncertain seasonal busyness
:width: 100%

The two fitted policies with uncertain seasonal busyness.
```

The `busyness` prior represents what the owner does not yet know about the season. The order distribution describes how the fitted rule chooses stock. When fitting and evaluating the policies, PyLIMID samples both possible demand conditions and the actions the rules choose.

## Next steps

- [Building a diagram](../user_guide/building_a_diagram.md) explains how to build and edit the shop's model, including what each decision knows and remembers.
