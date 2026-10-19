---
file_format: mystnb
kernelspec:
  name: python3
---

# The newsvendor

Every morning a street vendor must decide how many umbrellas to stock for the day. The weather is uncertain, and an umbrella not sold by evening is worthless. Stock too few and customers walk away; stock too many and the leftovers are a straight loss. Deciding *before* demand is known, with a per-unit cost on whichever side you get wrong, is the **newsvendor problem** — the classic single-period inventory decision, named for the original newspaper-stocking example.

This tutorial models it in `pylimid` and shows why the right answer is to **over-order**, well above the average demand. It is the library's mixed-type story in miniature: a categorical forecast, a continuous non-Gaussian demand, a discrete decision with a discrete information set, and a payoff that is an ordinary function.

## The numbers

The forecast is dry with probability 0.70 and rainy with probability 0.30, and demand is higher when it rains. Each umbrella costs \$1 wholesale and sells for \$10; leftovers are discarded.

Demand is positive and right-skewed, so it is modelled by a Gamma distribution with shape 4. The weather sets the mean, and shape 4 sets the spread at half the mean:

| Weather | Probability | Demand mean | Demand sd |
| --- | --- | --- | --- |
| dry | 0.70 | 40 | 20 |
| rainy | 0.30 | 120 | 60 |

The vendor orders from a discrete menu of stock levels (`0, 20, ..., 260`) and knows the forecast when ordering. So `weather` is a categorical chance node, `demand` is a continuous non-Gaussian chance node, and `order` is a discrete decision whose information set is `weather`. Demand itself is not an input to the decision — it is realized after the order is placed — which is exactly the shape the discrete-decision solvers require.

## Modelling it in pylimid

```{code-cell} ipython3
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve
```

```{code-cell} ipython3
COST = 1.0
PRICE = 10.0
WEATHER_STATES = ("dry", "rainy")
WEATHER_PROBS = jnp.array([0.70, 0.30])
DEMAND_MEANS = jnp.array([40.0, 120.0])
DEMAND_SHAPE = 4.0
ORDERS = jnp.arange(0, 261, 20)  # discrete stock levels
```

The two callables are the heart of the model. Demand is a Gamma whose rate is chosen so the mean follows the weather, and profit pays \$10 for every unit sold and \$1 for every unit ordered:

```{code-cell} ipython3
def demand_dist(weather):
    """Demand: a right-skewed Gamma whose mean is set by the weather."""
    return dist.Gamma(
        concentration=DEMAND_SHAPE, rate=DEMAND_SHAPE / DEMAND_MEANS[weather]
    )


def profit(order, demand):
    """Sell each unit for PRICE; pay COST for every unit ordered."""
    stocked = ORDERS[order]
    return PRICE * jnp.minimum(stocked, demand) - COST * stocked
```

With those pieces the diagram is a short, readable list of nodes. The decision carries `weather` as its parent — its information set:

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
print(diagram.validate())
```

A diagram is a mutable workspace, so inference is gated behind an explicit validation checkpoint. With a valid diagram, it also draws itself: circles for chance nodes, a rectangle for the decision, a diamond for the payoff.

```{code-cell} ipython3
diagram
```

Reading the arrows: `weather` feeds both the demand and the order, so the vendor sees the forecast before stocking; `order` and `demand` meet in the payoff.

## Seeing the uncertainty

The distributions are the whole story, so draw them. The dotted line is the mean and the dashed line is the 90th percentile, which we compute by inverting the CDF:

```{code-cell} ipython3
def demand_quantile(weather, p):
    """Invert the demand CDF by bisection (NumPyro's Gamma icdf needs TFP)."""
    low, high = 0.0, 2000.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if float(demand_dist(weather).cdf(mid)) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


critical = (PRICE - COST) / PRICE  # the 0.9 quantile, derived below
colors = ("#e9c46a", "#457b9d")

xs = jnp.linspace(0.0, 300.0, 500)
fig, ax = plt.subplots(figsize=(8, 4))
for index, (color, name) in enumerate(zip(colors, WEATHER_STATES, strict=True)):
    gamma = demand_dist(index)
    ax.plot(xs, jnp.exp(gamma.log_prob(xs)), color=color, label=f"{name} demand")
    ax.axvline(DEMAND_MEANS[index], color=color, linestyle=":", linewidth=1)
    q90 = demand_quantile(index, critical)
    ax.axvline(q90, color=color, linestyle="--", linewidth=1)
ax.set_xlabel("umbrellas demanded")
ax.set_ylabel("density")
ax.set_title("Demand by weather: mean (dotted) vs 90th percentile (dashed)")
ax.legend()
```

Both distributions are right-skewed: the mean sits left of the long right tail, and the 90th percentile sits well to its right — about 67 on a dry day and 200 on a rainy one, against means of 40 and 120.

## Solving it

`solve()` searches the policies and returns the best one with its expected utility. Because this diagram is solvable, the solver runs backward induction automatically, handling the continuous `demand` by Monte-Carlo:

```{code-cell} ipython3
solution = solve(diagram, num_samples=300_000)
print("method:", solution.method)
print("expected utility:", round(float(solution.expected_utility), 1))
for info, action in solution.policy["order"].items():
    print(f"  weather = {WEATHER_STATES[info[0]]:<6} -> order {int(ORDERS[action])}")
```

The strategy is to stock **60 on a dry day** and **200 on a rainy day** — in both cases far above the average demand. The rest of this tutorial is about why that is not a bug.

## Why we over-order

Ask what one extra umbrella is worth. It sells whenever demand reaches the stock level, with probability ``1 - F(Q)``, earning the \$9 margin. Otherwise it is left over, costing \$1. The best stock level equalizes the two:

$$
(1 - F(Q^*)) \cdot 9 = F(Q^*) \cdot 1
\quad\Longrightarrow\quad
F(Q^*) = \frac{\text{price} - \text{cost}}{\text{price}} = 0.9.
$$

The optimal stock is the **90th percentile** of demand, not the mean — the **critical fractile**. A missed sale costs nine times as much as a leftover, so it is worth carrying a lot of stock on the chance the day is busy.

The exact expected profit confirms it. For a continuous order size the expectation has a closed form in the Gamma CDF:

```{code-cell} ipython3
def expected_profit(order_qty, weather):
    """Exact E[profit] for a continuous order size, via the Gamma CDF."""
    cdf = demand_dist(weather).cdf(order_qty)
    cdf_plus = dist.Gamma(
        concentration=DEMAND_SHAPE + 1, rate=DEMAND_SHAPE / DEMAND_MEANS[weather]
    ).cdf(order_qty)
    return (
        PRICE * (DEMAND_MEANS[weather] * cdf_plus + order_qty * (1 - cdf))
        - COST * order_qty
    )


qs = jnp.linspace(0.0, 300.0, 300)
fig, ax = plt.subplots(figsize=(8, 4))
for index, (color, name) in enumerate(zip(colors, WEATHER_STATES, strict=True)):
    curve = jnp.array([expected_profit(q, index) for q in qs])
    ax.plot(qs, curve, color=color, label=f"{name} day")
    q90 = demand_quantile(index, critical)
    ax.axvline(DEMAND_MEANS[index], color=color, linestyle=":", linewidth=1)
    ax.plot([q90], [expected_profit(q90, index)], "o", color=color)
ax.set_xlabel("umbrellas ordered")
ax.set_ylabel("expected profit")
ax.set_title("Expected profit peaks at the 90th percentile, not the mean")
ax.legend()
```

Each curve peaks to the right of its dotted mean line, at the 90th percentile. Ordering the mean leaves money on the table:

```{code-cell} ipython3
for index, name in enumerate(WEATHER_STATES):
    q90 = demand_quantile(index, critical)
    at_mean = expected_profit(DEMAND_MEANS[index], index)
    at_q90 = expected_profit(q90, index)
    print(
        f"{name:<6}: mean {DEMAND_MEANS[index]:5.0f} -> {at_mean:6.1f}"
        f"   |   90th pct {q90:5.1f} -> {at_q90:6.1f}"
    )
```

The solver's grid picks land within one step of the exact fractiles (60 against 66.8 on a dry day, 200 against 200.4 on a rainy one), because the profit curve is nearly flat near its peak. The overall expected utility is about **508**, against **451** for ordering the mean every day.

```{admonition} The general rule
:class: tip
Whatever the demand distribution, the optimal stock solves ``F(Q) = (price - cost) / price``. Cheap goods with fat margins get stocked at a high quantile — far above the mean when demand is right-skewed — while expensive, low-margin goods get stocked below it. The distribution's *shape* matters, which is why a Gamma and not a Normal: a right-skewed demand has a longer way to run before the missed-sale risk is balanced.
```

```{admonition} Next
:class: seealso
[The oil field](oil_field.md) is a fully discrete decision problem with a memory arc, and [the logistics center](logistics_center.md) shows the other mixed-type pattern: a continuous score binned into a discrete report before the decision sees it. For inference without decisions, see [Bayesian networks](bayesian_networks.md).
```
