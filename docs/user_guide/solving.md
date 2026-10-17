---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving a diagram

Once we are happy with the way our influence diagram models the decision problem, we can use {py:func}`solve` to find the best policy.

This page shows how to read what it returns, how many samples we need to trust it, and how to check that the actions we offered are the right ones. It also explains how {py:func}`solve` finds the policy and when it fails.

We continue with the ice cream shop from the [Building a diagram](building_a_diagram.md) page:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

DEMAND_MEANS = jnp.array([40.0, 120.0])
BOOKINGS_BOOST = jnp.array([1.0, 1.5])
ORDERS = jnp.arange(0, 201, 20)


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
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("weather", "bookings"),
        dist=lambda weather, bookings: dist.Gamma(
            4.0, 4.0 / (DEMAND_MEANS[weather] * BOOKINGS_BOOST[bookings])
        ),
    )
)
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

diagram
```

## Reading the solution

{py:func}`solve` returns a {py:class}`Solution` object, which we can easily read with {py:meth}`~Solution.render`:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The policy tells the owner how many ice creams to order for each kind of weather.

A {py:class}`Solution` has three attributes:

- `policy` has one entry per decision. An entry maps a situation to the recommended action.
- `expected_utility` estimates the expected total utility when the shop follows this policy. Here, that is the shop's expected profit.
- `method` records which solver ran (more on this at the end of the page).

`render` is just a friendly way of reading `policy`. Inside it, a situation is a tuple with the state index of each of the decision's parents, and the action is an index into the decision's `states`. Here `order` has a single parent, so `(0,)` means `"cloudy"` and `(1,)` means `"sunny"`. If we ever need to work with the policy directly, we can decode it ourselves with the nodes' `states`:

```{code-cell} ipython3
for (weather,), action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[weather]}: stock {diagram['order'].states[action]}")
```

## How much can we trust the result?

PyLIMID uses sampling as its inference engine, so the result depends on how many random draws we take (`num_samples`) and the random key that generates them (`rng_key`). We can solve the diagram with a few different keys to see how much the answer depends on the draws:

```{code-cell} ipython3
for num_samples in (2_000, 50_000):
    for seed in range(4):
        result = solve(diagram, num_samples=num_samples, rng_key=jax.random.PRNGKey(seed))
        orders = [diagram["order"].states[a] for a in result.policy["order"].values()]
        print(f"{num_samples:>6} samples, seed {seed}: stock {orders}, profit {result.expected_utility:.1f}")
```

The policy is the same in every run, but the expected profit is not: with 2,000 samples it moves by more than 20 between keys, and with 50,000 by about 3. So if we want a precise expected utility, we need more samples.

A good habit before relying on a result is to solve it several times with different `rng_key` values. If the policy changes, either two actions are nearly as good as each other or `num_samples` is too small to tell them apart.

On a small diagram, compiling the model can take much of the runtime, so increasing `num_samples` does not necessarily increase the runtime in proportion.

## Are the actions the right ones?

The model recommends an order of 200 ice creams on a sunny day, which is the largest order on the menu. Now, when the best action sits at the edge of the action space like this, the true best may lie beyond it.

To check, we widen the action space up to 400 and solve again. The `profit` function reads `ORDERS` when it runs, so redefining `ORDERS` and the decision's `states` is enough:

```{code-cell} ipython3
ORDERS = jnp.arange(0, 401, 20)
diagram["order"].states = tuple(str(n) for n in ORDERS)

solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The best sunny-day order is now 240, so 200 was only the best option within the restricted action space we had defined. Unless the shop cannot stock more than 200, the owner should order 240 instead.

The action space also constrains us in the other direction. The solver only compares the options we list, so with a step of 20 it could never find an optimum of, let's say, 50.

## How does solve() find the policy?

So far we haven't had to think about how {py:func}`solve` finds the best policy, because the default, `method="auto"`, picks the solver method for us. There are two:

- `backward_induction` resolves one decision at a time, from the last to the first. It finds the best policy when the decisions can be resolved one after the other. PyLIMID checks this with {py:func}`~pylimid.inference.numpyro.solvers.is_solvable` when we select `auto`. A diagram with a single decision, like ours, always qualifies.
- `scan` follows a brute-force approach. It simulates every possible policy in the action space and keeps the best one. It works on any diagram but, since the number of policies grows exponentially with the number of situations a decision can face, it quickly becomes impractical. In our problem, `order` has 21 actions and 2 situations (`cloudy` and `sunny`), so there are $21^2 = 441$ policies, each simulated `num_samples` times. If the owner also had a report showing whether hotel bookings are normal or high for that day, there would be 4 situations and $21^4 = 194,481$ policies. If we add another two-state parent variable the scan is no longer an option, even though the diagram itself is still small.

### Example: when we can't use backward induction

The shop is doing well, so the owner buys two small carts to sell ice cream around town during the summer. The owner's children, Mia and Leo, each run a cart. Each of them is quite independent, so they decide where to set up, the beach or the park, without checking with the other:

- A day at the beach brings in \$300 and a day at the park \$200.
- If both carts go to the same place, they split its customers, so together they only make that spot's sales.
- Leo is too young to drive, so to get to the beach he has to pay \$20 for a van ride. The park is within walking distance for both.

This is a different decision from the ice cream shop problem, so we build a new diagram for it:

```{code-cell} ipython3
SALES = jnp.array([300.0, 200.0])  # beach, park
LEO_TRAVEL = jnp.array([20.0, 0.0])


def carts_profit(mia_spot, leo_spot):
    same_spot = mia_spot == leo_spot
    sales = jnp.where(same_spot, SALES[mia_spot], SALES[mia_spot] + SALES[leo_spot])
    return sales - LEO_TRAVEL[leo_spot]


carts = InfluenceDiagram()
carts.add_node(DecisionNode(name="mia_spot", states=("beach", "park")))
carts.add_node(DecisionNode(name="leo_spot", states=("beach", "park")))
carts.add_node(
    UtilityNode(name="profit", parents=("mia_spot", "leo_spot"), values=carts_profit)
)
carts
```

Leo's best spot depends on where Mia decides to go, and the other way around: if Mia takes the beach, Leo does better at the park, and if Mia takes the park, Leo does better at the beach. But neither sees the other's choice, so neither decision can be resolved first. As a result, backward induction can't be used:

```{code-cell} ipython3
from pylimid.inference.numpyro.solvers import is_solvable

is_solvable(carts.snapshot())
```

The scan has no trouble here since each decision has two actions and there is a single situation, so there are only $2 \times 2 = 4$ policies to try:

```{code-cell} ipython3
print(solve(carts).render(carts))
```

Mia should take the beach and Leo the park. Note that the reason backward induction fails is that each decision's best option depends on a choice the other can't see, **which leaves no order in which to resolve them**.

## When does solve() fail?

{py:func}`solve` raises an {py:class}`~pylimid.InferenceError` in these cases:

- **The diagram has no decisions.** In that case there is nothing to optimise, so there is no policy to return.
- **A decision observes a continuous variable.** A policy is a table with one action per situation, so a decision's parents must be discrete chance or decision nodes. If we want it to react to something continuous, like a temperature reading, we add an intermediate discrete chance node that bins it (say, cold, mild and hot) as the parent.
- **A situation never comes up in the simulation.** Backward induction needs at least one draw for every combination of a decision's parent states. If a situation is rare, increasing `num_samples` can help. If it is impossible (its probability is 0), more samples cannot help. In that case, we can use `method="scan"` instead. The scan evaluates complete policies, so the action for an impossible situation does not affect the payoff. Such situations still increase the number of policies it needs to try though.
