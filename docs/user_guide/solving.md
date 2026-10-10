---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving a diagram

Once we are happy with the way our influence diagram models the decision problem, finding the optimal solution to it can be done with {py:func}`solve`.

This page shows how to read what it returns, how much to trust it, and how to use it to answer two questions that come up in almost every decision model: are the actions on offer the right ones, and what is a piece of information worth?

We continue with the vendor problem from the [Building a diagram](building_a_diagram.md) page:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

DEMAND_MEANS = jnp.array([40.0, 120.0])
WEEKEND_BOOST = jnp.array([1.0, 1.5])
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
        name="weekend",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([5 / 7, 2 / 7])),
    )
)
diagram.add_node(
    ChanceNode(
        name="demand",
        parents=("weather", "weekend"),
        dist=lambda weather, weekend: dist.Gamma(
            4.0, 4.0 / (DEMAND_MEANS[weather] * WEEKEND_BOOST[weekend])
        ),
    )
)
diagram.add_node(
    DecisionNode(
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))
```

## Reading the solution

{py:func}`solve` returns a {py:class}`Solution` object:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
solution
```

`policy` has one entry per decision. Each entry maps a situation the decision can face to the action to take in it. A situation is a tuple with the state index of each parent, in the order the decision lists its parents, and the action is an index into the decision's `states`. Here `order` has a single parent, so `(0,)` means cloudy and `(1,)` means sunny. The node labels turn the policy into something readable:

```{code-cell} ipython3
for (weather,), action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[weather]}: stock {diagram['order'].states[action]}")
```

`expected_utility` is the expected profit when the vendor follows this policy, and `method` records which solver ran (more on this at the end of the page).

## How much to trust the numbers

{py:func}`solve` estimates expected utilities by simulating the diagram `num_samples` times (2,000 by default). The simulation uses a fixed random key, so the same call always returns the same answer, but a different key gives a slightly different one. Solving with a few keys shows how much the answer depends on the simulation:

```{code-cell} ipython3
for num_samples in (2_000, 50_000):
    for seed in range(4):
        result = solve(diagram, num_samples=num_samples, rng_key=jax.random.PRNGKey(seed))
        orders = [diagram["order"].states[a] for a in result.policy["order"].values()]
        print(f"{num_samples:>6} samples, seed {seed}: stock {orders}, profit {result.expected_utility:.1f}")
```

The policy is the same in every run, but the expected profit is not: with 2,000 samples it moves by more than 20 between keys, and with 50,000 by about 3. This suggests a simple routine: before relying on a result, solve again with another `rng_key`. If the policy changes, two actions are nearly as good as each other, or `num_samples` is too small to tell them apart.

More samples cost less than you might expect. Each call to {py:func}`solve` compiles the model before it simulates, and on a diagram this size the compilation takes most of the time, so 50,000 samples take about as long as 2,000.

## Are the actions the right ones?

On a sunny day the policy stocks 200 ice creams, which is the largest order on the menu. When the best action is at the edge of what is on offer, the real best may lie beyond it. To check, we widen the menu up to 400 and solve again. `profit` reads `ORDERS` when it runs, so redefining `ORDERS` and the decision's `states` is enough:

```{code-cell} ipython3
ORDERS = jnp.arange(0, 401, 20)
diagram["order"].states = tuple(str(n) for n in ORDERS)

solution = solve(diagram, num_samples=50_000)
for (weather,), action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[weather]}: stock {diagram['order'].states[action]}")
print(f"expected profit: {solution.expected_utility:.1f}")
```

The best sunny-day order is now 240, inside the menu, so the result is no longer limited by the actions we offered. The menu works the other way too: the solver only compares the actions you list, so a step of 20 can't find an optimum of 50.

## What is a piece of information worth?

A decision knows only its parents. `order` sees the forecast, but not whether it is a weekend, even though demand depends on it. Adding an arc from `weekend` to `order` lets the vendor use that information, and the gain in expected profit is what knowing it is worth.

Because each estimate has some noise, compare the two diagrams with the same `rng_key`. Both are then simulated with the same random draws, so the noise largely cancels in the difference:

```{code-cell} ipython3
def expected_profit(seed):
    key = jax.random.PRNGKey(seed)
    return solve(diagram, num_samples=50_000, rng_key=key).expected_utility


without_weekend = [expected_profit(seed) for seed in range(3)]
diagram.add_arc("weekend", "order")
with_weekend = [expected_profit(seed) for seed in range(3)]

for seed, (before, after) in enumerate(zip(without_weekend, with_weekend)):
    print(f"seed {seed}: {before:.1f} -> {after:.1f}, gain {after - before:.1f}")
```

Each estimate moves by a few dollars between keys, but the gain barely moves. With the new arc, the policy has one entry per combination of forecast and day, in the order of `order`'s parents:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
for (weather, weekend), action in solution.policy["order"].items():
    print(
        f"{diagram['weather'].states[weather]}, weekend {diagram['weekend'].states[weekend]}: "
        f"stock {diagram['order'].states[action]}"
    )
```

The same approach prices everything the vendor knows. Removing both arcs leaves the vendor deciding blind, with a single action for every day:

```{code-cell} ipython3
diagram.remove_arc("weather", "order")
diagram.remove_arc("weekend", "order")

blind = solve(diagram, num_samples=50_000)
print(f"stock {diagram['order'].states[blind.policy['order'][()]]} every day")
print(f"value of the forecast and the day: {with_weekend[0] - blind.expected_utility:.1f}")
```

The policy for a decision with no parents has a single entry, keyed by the empty tuple `()`. The difference is the most the vendor should pay, per day, to know the forecast and the day of the week before ordering.

```{admonition} Decisions don't remember
:class: note

With several decisions, a later decision also knows only its parents. It doesn't automatically know what an earlier decision chose or what that decision saw; this is the "limited memory" in *limited-memory influence diagram*. If a later decision should know them, add the arcs. [The oil field](../tutorials/oil_field.md) tutorial has a diagram with two decisions where this matters.
```

## Which solver runs

PyLIMID has two solvers, and `method="auto"`, the default, picks between them:

- **Backward induction** resolves one decision at a time, starting with the last one. It is fast, but it only finds the best policy on diagrams where the decisions can be resolved one at a time; {py:func}`~pylimid.inference.numpyro.solvers.is_solvable` checks this, and `auto` uses backward induction whenever it holds. A diagram with a single decision always qualifies.
- **The scan** tries every policy and keeps the best. It works on any diagram, but the number of policies grows very quickly: with 21 possible orders and 4 combinations of forecast and day, there are 21⁴ = 194,481 of them, each simulated `num_samples` times.

{py:attr}`Solution.method` tells you which one ran. Passing `method="scan"` on a diagram that backward induction can solve is a way to cross-check a result on a small model; passing `method="backward_induction"` on a diagram it can't solve raises an error.

## When solve() refuses

{py:func}`solve` raises an {py:class}`~pylimid.inference.engine.InferenceError` in these cases:

- **The diagram has no decisions.** Use {py:func}`infer` to compute posteriors instead.
- **A decision has a continuous parent.** A policy is a table with one action per situation, so everything a decision observes must be discrete. To let a decision react to a continuous quantity, such as a temperature reading, add a discrete chance node that bins it and make that node the parent.
- **A situation never comes up in the simulation.** Every combination of the parents' states needs at least one simulated sample. A rare combination can be missed with too few samples, and the error asks you to increase `num_samples`. An impossible one, such as a state that can never happen, needs to be removed from the model.
