---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple decisions

So far, the shop's owner has faced a single choice: how many ice creams to order. Now, suppose the owner is considering a daily promotion that costs \$80 but is expected to increase average demand by 20%. However, it must be announced the evening before, when the next day's weather is still uncertain.

There are now two decisions. First, the owner must choose whether to run the promotion or not. Then, in the morning (when the weather forecast is known), the owner must decide how much stock to order.

This page shows how to represent those choices, read their policies and explore the resulting profits. We keep the weather, bookings and Gamma demand from the previous pages, with the wider order menu from the [Solving](solving.md) guide. Bookings remain unknown at both decisions.

## Adding the promotion decision

The new {py:class}`DecisionNode`, `promotion`, has two actions, `skip` and `run`. It has no parents because the owner has neither the next day's weather nor the hotel bookings information when setting up the promotion.

We add `promotion` as a parent of `demand` so the promotion can increase demand, and as a parent of `order` so the owner can take the earlier choice into account:

```{code-cell} ipython3
from copy import deepcopy

import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
    infer,
    solve,
)

DEMAND_MEANS = jnp.array([40.0, 120.0])
BOOKINGS_BOOST = jnp.array([1.0, 1.5])
PROMOTION_BOOST = jnp.array([1.0, 1.2])
PROMOTION_COST = 80.0
ORDERS = jnp.arange(0, 401, 20)


def demand_dist(weather, bookings, promotion):
    mean = DEMAND_MEANS[weather] * BOOKINGS_BOOST[bookings] * PROMOTION_BOOST[promotion]
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
        name="bookings",
        states=("normal", "high"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_node(DecisionNode(name="promotion", states=("skip", "run")))
diagram.add_node(
    ChanceNode(
        name="demand", parents=("weather", "bookings", "promotion"), dist=demand_dist
    )
)
diagram.add_node(
    DecisionNode(
        name="order",
        parents=("weather", "promotion"),
        states=tuple(str(n) for n in ORDERS),
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

Notice that the two arcs from `promotion` have different meanings. For `demand`, the parent changes the distribution. For `order`, it says that the owner remembers the earlier choice. The weather is only a parent of `order`, so the promotion cannot be chosen in response to tomorrow's weather.

## Reading the two policies

We can solve this diagram with the same {py:func}`solve` call as before:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The owner should always run the promotion. After that choice, the recommended order depends on the weather: 100 ice creams on a cloudy day and 280 on a sunny day. The estimated total profit is about \$609.

`solution.policy` has one entry for each decision. The `promotion` entry has the single key `()`, because it has no parents. The `order` keys contain two state indices, following its parent order: `(weather, promotion)`. Thus, `(1, 1)` means sunny weather and a promotion that runs.

The order policy also contains rows for a skipped promotion because backward induction considers how the owner would order under either promotion choice before deciding which one is better. Those rows remain in the policy **even though the recommended promotion does not use them**.

## Why memory must be explicit

Suppose we remove `promotion` from `order` parents:

```{code-cell} ipython3
forgetful_diagram = deepcopy(diagram)
forgetful_diagram.remove_arc("promotion", "order")
forgetful_diagram["order"].parents
```

The promotion still affects demand and still costs money. But the order decision can now only react to the weather, using the same rule whether the promotion ran or not. **This is a different decision problem**. PyLIMID does not automatically make an earlier choice available to a later decision.

This change also affects the solver. We can check whether each diagram has an order in which backward induction can resolve its decisions:

```{code-cell} ipython3
from pylimid.inference.numpyro.solvers import is_solvable

print("with memory:", is_solvable(diagram.snapshot()))
print("without memory:", is_solvable(forgetful_diagram.snapshot()))
```

The original diagram qualifies, but the version without memory does not. In that version, `solve(method="auto")` would select the scan, which evaluates every complete policy instead of resolving the decisions in turn.

A later decision also needs an explicit parent for each earlier observation it should still use. **Information is not inherited through the earlier decision node**. The parents describe the information available at each choice, so they should reflect what the decision maker actually knows.

## Exploring the outcomes

We can use {py:func}`infer` to estimate the total profit under the recommended policies, adding the means of both utility nodes:

```{code-cell} ipython3
outcomes = infer(
    diagram,
    query=["profit", "promotion_cost"],
    policy=solution.policy,
    num_samples=50_000,
)
round(outcomes["profit"].mean() + outcomes["promotion_cost"].mean(), 1)
```

The estimated average total profit is about \$605 per day, including the promotion cost.

```{admonition} Reminder: Monte Carlo estimates
:class: info

Both calls use the same policies. The estimates differ slightly because `solve` and `infer` use different simulated days.
```
