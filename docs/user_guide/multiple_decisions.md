---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple decisions

So far, the shop's owner has faced a single choice: how many ice creams to order. Now, suppose the owner is considering a daily promotion that costs \$80 but is expected to increase average demand by 20%. However, it must be announced the evening before, when the next day's weather is still uncertain.

There are now two decisions. First, the owner must choose whether to run the promotion or not. Then, in the morning (when the weather is known), the owner must decide how much stock to order.

This page shows how to represent those choices, read their policies and explore the resulting profits. We keep the weather, `busyness` prior, bookings and {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` demand from the previous pages, with the wider order menu from the [Solving](solving.md) guide. Bookings and the actual `busyness` remain unknown at both decisions.

## Adding the promotion decision

The new {py:class}`DecisionNode`, `promotion`, has two actions, `skip` and `run`. It has no parents because neither tomorrow's weather nor hotel bookings are known when the owner makes this choice.

We add `promotion` as a parent of `demand` so the promotion can increase demand, and as a parent of `order` so the owner can take the earlier choice into account. Both decisions use {py:class}`TabularSoftmax` to fit action probabilities for each situation the owner can observe:

```{code-cell} ipython3
from copy import deepcopy

import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    TabularSoftmax,
    UtilityNode,
    infer,
    solve,
)

DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])
PROMOTION_BOOST = jnp.array([1.0, 1.2])
PROMOTION_COST = 80.0
ORDERS = jnp.arange(0, 401, 20)


def demand_dist(weather, busyness, bookings, promotion):
    mean = (
        DEMAND_MEANS[weather]
        * busyness
        * BOOKINGS_BOOST[bookings]
        * PROMOTION_BOOST[promotion]
    )
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - 1.0 * stocked


diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="weather",
        states=("cloudy", "sunny"),
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
    ChanceNode(
        name="bookings",
        states=("normal", "high"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_node(
    DecisionNode(
        name="promotion", states=("skip", "run"), policy_family=TabularSoftmax()
    )
)
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("weather", "busyness", "bookings", "promotion"),
        dist=demand_dist,
    )
)
diagram.add_node(
    DecisionNode(
        name="order",
        parents=("weather", "promotion"),
        states=tuple(str(n) for n in ORDERS),
        policy_family=TabularSoftmax(),
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
diagram.add_node(
    UtilityNode(
        name="promotion_cost",
        parents=("promotion",),
        values=lambda promotion: -PROMOTION_COST * promotion,
    )
)
diagram
```

We put the promotion cost in a separate utility node, so the total payoff is their sum, as explained in [Distributions and utilities](distributions_and_utilities.md). The cost is charged whenever the promotion runs, regardless of how many ice creams the shop sells.

Notice that the two arcs from `promotion` have different meanings. For `demand`, the parent changes the distribution. For `order`, it says that the owner remembers the earlier choice. Of the two decisions, only `order` has weather as a parent, so the promotion cannot be chosen in response to tomorrow's weather.

## Reading the two policies

We use the same {py:func}`solve` call as before. It fits both policies together, using the total profit from the two utility nodes:

```{code-cell} ipython3
solution = solve(diagram)
print(solution.render())
```

The fitted policy almost always runs the promotion. When it runs, the most likely orders are 100 ice creams on a cloudy day and 320 on a sunny day. The estimated total profit is about \$671.9. The rules still sample their actions, and the displayed probabilities are rounded.

`solution.policy` contains one fitted {py:class}`CategoricalPolicy` for each decision. The `promotion` rule has no parents, so its probabilities for `skip` and `run` apply every day. The `order` rule has a row for each combination of weather and the earlier promotion choice. For example, `weather=1, promotion=1` means sunny weather and a promotion that runs.

`TabularSoftmax` also shows order rows for days when the promotion is skipped. These rows receive training only when that situation occurs during simulation. Once skipping becomes rare, they receive little further training. We should read their probabilities with the same caution as the rarely visited rows discussed in [Solving](solving.md#what-if-a-situation-never-appears).

## Why memory must be explicit

Suppose we remove `promotion` from the parents of `order`:

```{code-cell} ipython3
forgetful_diagram = deepcopy(diagram)
forgetful_diagram.remove_arc("promotion", "order")
forgetful_diagram["order"].parents
```

The promotion still affects demand and still costs money. But the order decision can now only react to the weather, using the same rule whether the promotion ran or not. **This is a different decision problem**. PyLIMID does not automatically make an earlier choice available to a later decision.

We can solve this version too. The order rule now has one row for each kind of weather:

```{code-cell} ipython3
forgetful_solution = solve(forgetful_diagram)
print(forgetful_solution.render())
```

Both versions favour running the promotion and have the same most likely orders in this run. When the promotion is skipped, however, the version without memory must still use the same weather-based order probabilities.

A later decision also needs an explicit parent for each earlier observation it should still use. **Information is not inherited through the earlier decision node**. The parents should reflect what the owner actually knows at each choice.

## Exploring the outcomes

We return to the diagram where the owner remembers the promotion choice. We use {py:func}`infer` to explore the total profit under its fitted policies, adding the draws from both utility nodes:

```{code-cell} ipython3
outcomes = infer(
    diagram,
    query=["profit", "promotion_cost"],
    policy=solution.policy,
    num_samples=50_000,
)
total_profit = outcomes["profit"].values + outcomes["promotion_cost"].values
round(total_profit.mean().item(), 1)
```

Values at the same position in the two arrays belong to the same simulated day. Adding them gives that day's total profit, including the promotion cost. The estimated average is about \$677.5 per day.

```{admonition} Reminder: Monte Carlo estimates
:class: info

Here, `infer` evaluates `solution.policy`, the policy from the original diagram. The average above can differ from `solution.expected_utility` because the two estimates use different simulated days.
```
