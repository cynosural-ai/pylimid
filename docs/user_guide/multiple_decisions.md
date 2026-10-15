---
file_format: mystnb
kernelspec:
  name: python3
---

# Multiple decisions

So far, the shop's owner has made one choice: how many ice creams to order. Suppose the shop is now considering a local promotion. It costs \$80 and is expected to increase average demand by 20%, but it must be booked the evening before, when the next day's weather is still unknown.

There are now two decisions. First, the owner chooses whether to run the promotion. Then, in the morning, the owner knows the weather and decides how much stock to order, remembering whether the promotion was booked.

This page shows how to represent those choices, give the order decision a hotel report, and read the resulting policies. We keep the weather, bookings and Gamma demand from the previous pages, with the wider order menu from [Solving](solving.md).

## Adding the promotion decision

The new {py:class}`DecisionNode`, `promotion`, has two actions, skip and run. It has no parents because the owner has neither the next day's weather nor current hotel bookings when booking the promotion. As before, decisions choose from a finite set of actions and can only observe discrete variables.

We add `promotion` as a parent of `demand` so the promotion can increase demand, and as a parent of `order` so the owner can take the earlier choice into account:

```{code-cell} ipython3
from copy import deepcopy

import jax
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

The `profit` function still describes revenue minus stock cost. We put the promotion cost in a separate utility node, so the total payoff is their sum, as explained in [Distributions and utilities](distributions_and_utilities.md). The cost is charged whenever the promotion runs, regardless of how many ice creams the shop sells.

Notice that the two arcs from `promotion` have different meanings. For `demand`, the parent changes the distribution. For `order`, it says that the owner remembers the earlier choice. The weather is only a parent of `order`, so the promotion cannot be chosen in response to tomorrow's weather.

## Giving the order decision a report

In [Inference](inference.md), the owner obtained a report that accurately identified the booking state. Suppose the report is instead a noisy numerical score, available in the morning at no cost. Its average is 0 when bookings are normal and 1 when they are high, with a standard deviation of 0.5 in either case.

The score is continuous, so it cannot be a direct parent of `order`. We represent a decision rule that uses two categories: a score below 0.5 means a normal-bookings report, and a score of 0.5 or above means a high-bookings report:

```{code-cell} ipython3
def report_dist(booking_score):
    state = jnp.where(booking_score < 0.5, 0, 1)
    return dist.Categorical(probs=jax.nn.one_hot(state, 2))


diagram.add_node(
    ChanceNode(
        name="booking_score",
        parents=("bookings",),
        dist=lambda bookings: dist.Normal(loc=bookings, scale=0.5),
    )
)
diagram.add_node(
    ChanceNode(
        name="report",
        parents=("booking_score",),
        states=("normal", "high"),
        dist=report_dist,
    )
)
diagram.add_arc("report", "order")
diagram
```

`jax.nn.one_hot` produces the probability vector `[1, 0]` or `[0, 1]`, so the report category is determined by the score. The uncertainty comes from the score itself: high bookings can produce a low score, and normal bookings can produce a high one.

The order decision now observes `weather`, `promotion` and `report`, in that order. It does not observe the true booking state or use the precise numerical score. Binning the score lets us use it in a finite policy table, at the cost of treating every score in a category alike.

## Reading the two policies

We can solve this diagram with the same {py:func}`solve` call as before:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The owner should run the promotion, with an estimated total profit of about \$617. After that choice, the recommended order depends on the weather and the report. For example, a sunny day with a high-bookings report calls for 340 ice creams in this run.

`solution.policy` has one entry for each decision. The `promotion` entry has the single key `()`, because it has no parents. The `order` keys contain three state indices, following its parent order: `(weather, promotion, report)`. Thus, `(1, 1, 1)` means sunny weather, a promotion that runs, and a high-bookings report.

The order policy also contains rows for a skipped promotion. Backward induction considers how the owner would order under either promotion choice before deciding which choice is better. Those rows remain in the policy even though the recommended promotion does not use them.

Having two decisions does not, by itself, require a different solver. Here backward induction finds the order policy first, then evaluates whether running the promotion is worthwhile when the shop follows that policy. The reported total profit includes the promotion cost. As on the [Solving](solving.md) page, the policy and expected utility are Monte Carlo estimates; nearby order quantities can change when we use different random keys.

## Why memory must be explicit

Suppose we remove `promotion` from the order decision's parents:

```{code-cell} ipython3
forgetful_diagram = deepcopy(diagram)
forgetful_diagram.remove_arc("promotion", "order")
forgetful_diagram["order"].parents
```

The promotion still affects demand and still costs money. But the order decision can now only react to the weather and report, using the same rule whether the promotion ran or not. This is a different decision problem: PyLIMID does not automatically make an earlier choice available to a later decision.

This change also affects the solver. We can check whether each diagram has an order in which backward induction can resolve its decisions:

```{code-cell} ipython3
from pylimid.inference.numpyro.solvers import is_solvable

print("with memory:", is_solvable(diagram.snapshot()))
print("without memory:", is_solvable(forgetful_diagram.snapshot()))
```

The original diagram qualifies, but the version without memory does not. In that version, `solve(method="auto")` would select the scan. Although it has fewer order-policy rows, trying every complete policy is expensive with this menu, so we do not run the scan here. See [Solving](solving.md) for how its cost grows.

A later decision also needs an explicit parent for each earlier observation it should still use. Information is not inherited through the earlier decision node. The parents describe the information available at each choice, so they should reflect what the decision maker actually knows.

## Exploring the outcomes

We can use {py:func}`infer` to explore the outcomes of both decisions. Passing `solution.policy` supplies both rules, so each simulated day includes the promotion choice and an order based on the weather and report:

```{code-cell} ipython3
outcomes = infer(
    diagram,
    query=["profit", "promotion_cost"],
    policy=solution.policy,
    num_samples=50_000,
)
round(outcomes["profit"].mean() + outcomes["promotion_cost"].mean(), 1)
```

The estimated average total profit is about \$613. `infer` returns each utility separately, so we add their means to include the promotion cost. The small difference from `solution.expected_utility` comes from using different simulated days, as in the previous page.

Both decisions influence these outcomes, so both need a rule in `policy`. We can also substitute a fixed action for either decision to explore an alternative, just as we did for the original order policy in [Inference](inference.md).
