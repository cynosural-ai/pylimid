---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving a diagram

Once we are happy with the way our influence diagram models the decision problem, we can use {py:func}`solve` to fit policies that aim to maximise expected profit.

This page shows how to read both kinds of fitted rules, how to assess the result, and how the choice of policy family affects what we find. It also explains how `solve` fits the policies together and when it fails.

We continue with the shop from [Distributions and utilities](distributions_and_utilities.md). The owner decides whether to promote the evening before and orders stock after checking the morning weather. We keep the three utility nodes for revenue, stock cost and promotion cost:

```{code-cell} ipython3
import jax
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
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])


def demand_dist(weather, promotion, busyness, bookings):
    mean = (
        DEMAND_MEANS[weather]
        * PROMOTION_BOOST[promotion]
        * busyness
        * BOOKINGS_BOOST[bookings]
    )
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


def revenue(order, demand):
    return PRICE * jnp.minimum(order, demand)


def stock_cost(order):
    return -COST * order


def promotion_cost(promotion):
    return -PROMOTION_COST * promotion


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
    ChanceNode(
        name="demand",
        parents=("weather", "promotion", "busyness", "bookings"),
        dist=demand_dist,
    )
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
    UtilityNode(name="revenue", parents=("order", "demand"), values=revenue)
)
diagram.add_node(
    UtilityNode(name="stock_cost", parents=("order",), values=stock_cost)
)
diagram.add_node(
    UtilityNode(name="promotion_cost", parents=("promotion",), values=promotion_cost)
)

diagram
```

## Reading the solution

{py:func}`solve` returns a {py:class}`Solution` object, which we can read with {py:meth}`~Solution.render`. We use the same training budget as in the quickstart:

```{code-cell} ipython3
solution = solve(diagram, num_steps=2_000)
print(solution.render())
```

A `Solution` has two attributes:

- `policy` contains a fitted rule for each decision. Here, `promotion` has a {py:class}`CategoricalPolicy` and `order` has a {py:class}`ContinuousPolicy`. Each rule keeps its family, fitted parameters and information parents, together with its action labels or continuous domain.
- `expected_utility` estimates the expected total utility when the shop follows both rules. Here, that is the sum of revenue, stock cost and promotion cost, i.e., expected profit.

For the promotion, we can read the probabilities in the order of its states, `skip` and `run`:

```{code-cell} ipython3
promotion_rule = solution.policy["promotion"]
for label, probability in zip(
    promotion_rule.states, promotion_rule.probabilities(), strict=True
):
    print(f"{label}: {probability:.3%}")
```

The order rule returns a distribution over kilograms. We can inspect its mean for each combination of weather and promotion:

```{code-cell} ipython3
order_rule = solution.policy["order"]
for promotion, promotion_label in enumerate(promotion_rule.states):
    for weather, weather_label in enumerate(WEATHER_STATES):
        orders = order_rule.distribution(
            weather=jnp.array(weather), promotion=jnp.array(promotion)
        )
        print(f"{promotion_label}, {weather_label}: mean order {orders.mean:.1f} kg")
```

In this run, the fitted rule puts more than 99.9% probability on skipping the promotion. When it is skipped, the mean orders are about 7.6 kilograms on cloudy days and 23.7 on sunny days. The estimated expected profit is about \$623.8 per day. We still need to assess how well these policies have trained before drawing conclusions about the promotion.

Both rules remain stochastic. We can follow them by sampling the promotion, then sampling an order once the weather is known. Suppose tomorrow is sunny:

```{code-cell} ipython3
promotion_key, order_key = jax.random.split(jax.random.PRNGKey(7))
chosen_promotion = promotion_rule.sample(promotion_key)
chosen_order = order_rule.sample(
    order_key, weather=jnp.array(1), promotion=chosen_promotion
)
print("Promotion index:", chosen_promotion)
print(f"Order: {chosen_order:.2f} kg")
```

The sampled promotion does not use the weather, because the owner chooses it the evening before. The sampled order can use both the weather and that earlier choice. Always choosing the most likely promotion and ordering the mean would define a different policy, whose profit would need a separate evaluation.

## How much can we trust the result?

Training depends on the simulated days and on how long we let it run. `num_samples` sets the number of scenarios per update and in the final utility estimate, while `num_steps` sets the number of updates. The defaults are 10,000 samples and 500 updates. `learning_rate` controls the Adam step size and defaults to 0.05.

We can compare a shorter run and a different random key with the initial fit:

```{code-cell} ipython3
for steps, seed in ((500, 0), (2_000, 1)):
    result = solve(diagram, num_steps=steps, rng_key=jax.random.PRNGKey(seed))
    print(f"updates={steps}, seed={seed}")
    print(f"probability of promotion: {result.policy['promotion'].probabilities()[1]:.3%}")
    for weather, label in enumerate(WEATHER_STATES):
        orders = result.policy["order"].distribution(
            weather=jnp.array(weather), promotion=jnp.array(0)
        )
        print(f"{label}, promotion skipped: mean order {orders.mean:.1f} kg")
    print(f"expected profit: {result.expected_utility:.1f}")
```

The shorter run estimates profit at about \$621.6 per day, while the second random key gives about \$637.4 after 2,000 updates. All three fits almost always skip the promotion, but their sunny-day mean orders range from about 21.1 to 25.7 kilograms. Agreement about skipping does not tell us whether better promotion and order rules exist.

```{admonition} Choosing a training budget
:class: info

More samples, more updates and different random keys help in different ways:

- Increase `num_samples` to reduce noise in each training gradient estimate. This also makes the final expected-utility estimate more precise.
- Increase `num_steps` to give the optimizer more opportunities to improve the policies. More updates do not guarantee a better result.
- Try several `rng_key` values to check how consistently training finds similar policies and obtain candidates to compare.

The displayed profit is an estimate from fresh simulated days. Small differences may reflect estimation noise. We can evaluate candidate policies again with {py:func}`infer`, using more draws to distinguish differences that matter for the decision. More evaluation draws improve our estimate of an existing policy's utility. Improving the policy itself requires training or changing the family.
```

### What if a situation rarely appears?

In the initial fit, the promotion almost never runs, so the order policy receives little training for promoted days. It can still return distributions for those situations, but we should be cautious about interpreting them. The cost of the promotion is immediate, while its benefit also depends on ordering enough stock to serve the extra customers.

The {py:class}`Linear` model shares parameters across weather and promotion values. It does not fit an independent location for every combination, so training on one situation can affect another. Rare situations still provide little direct evidence about how their orders should change.

For a {py:class}`TabularSoftmax` family with information parents, each combination has a separate row. Rarely visited rows receive little training. A combination with zero probability does not affect expected utility, and more samples cannot make it appear.

## Are we fitting the right kind of rule?

The action domain and the policy family both matter. The shop can order any quantity between zero and 40 kilograms, but our family also fixes the underlying Normal's scale at 0.25 kilograms. More training cannot change that spread.

Suppose we allow more variation in the orders by fixing the scale at 1 kilogram instead. We keep the same prices, demand assumptions, information and freezer capacity, then fit both decisions again:

```{code-cell} ipython3
initial_solution = solution
diagram["order"].policy_family = DistributionPolicy(
    distribution=dist.TruncatedNormal,
    parameter_model=Linear(outputs=("loc",)),
    fixed_parameters={"low": 0.0, "high": CAPACITY, "scale": 1.0},
)

solution = solve(diagram, num_steps=2_000)
print(solution.render())
for weather, label in enumerate(WEATHER_STATES):
    orders = solution.policy["order"].distribution(
        weather=jnp.array(weather), promotion=jnp.array(1)
    )
    print(f"{label}, promotion running: mean order {orders.mean:.1f} kg")
```

With this family, the fitted rule almost always runs the promotion. When it runs, the mean orders are about 10.1 kilograms on cloudy days and 29.5 on sunny days. The estimated expected profit is about \$689.2 per day, compared with \$623.8 for the initial fit. The earlier preference for skipping changes when we fit a different order family.

```{figure} /_static/figures/solving/policy_comparison.png
:alt: The initial fit mostly skips the promotion and has narrow order distributions. The fit with a scale of one kilogram mostly runs the promotion and orders more on both cloudy and sunny days.
:width: 100%

The two fitted policies. Each pair of order curves shows the situation favoured by that fit's promotion rule.
```

Wider order distributions are not inherently more profitable. They change the orders explored during training as well as the policy we eventually follow. In this run, the wider family leads to a fit with higher estimated profit. We keep this family and the new `solution` for the rest of the example.

We could also learn the spread by using `Linear()` and omitting `scale` from `fixed_parameters`. That allows both location and scale to depend on the information parents, while keeping the truncation bounds fixed. Learning a spread towards zero can make the training gradients noisy. Choosing a suitable family remains part of modelling the decision problem. The [policy reference](../api/policy.rst) describes the available composition options.

### How much do the orders vary?

For a fitted {py:class}`ContinuousPolicy`, `distribution` returns the conditional NumPyro distribution. We can ask it for the central 95% of orders on a sunny day when the promotion runs:

```{code-cell} ipython3
sunny_orders = solution.policy["order"].distribution(
    weather=jnp.array(1), promotion=jnp.array(1)
)
lower, upper = sunny_orders.icdf(jnp.array([0.025, 0.975]))
print(f"Central 95% of sunny-day orders: {lower:.1f} to {upper:.1f} kg")
print(f"Probability of ordering above 39 kg: {1 - sunny_orders.cdf(39.0):.3%}")
```

The sunny-day rule places the central 95% of orders between about 27.5 and 31.5 kilograms, well below the freezer's capacity. This interval describes how the fitted policy varies its orders for that situation. Demand uncertainty is represented by the demand distribution, and estimation noise in expected profit is a separate question.

If the fitted orders concentrate near the freezer's limit, we should check whether that capacity is the intended constraint. Increasing the allowed domain does not automatically widen a policy distribution. Modelling a different capacity requires considering both the action domain and the distribution's bounds.

### Continuous actions and continuous information

The shop's `order` is a continuous decision, but its information parents, `weather` and `promotion`, are categorical. The kind of action and the kind of information are separate choices:

| Family | Actions | Information parents |
| --- | --- | --- |
| {py:class}`TabularSoftmax` | Categorical | Scalar categorical values |
| {py:class}`LinearSoftmax` | Categorical | Categorical and numerical values |
| {py:class}`DistributionPolicy` with {py:class}`Linear` | Scalar continuous | Categorical and numerical values |

For example, if the order decision receives a numerical forecast as an additional parent, `Linear` can use it. A categorical decision with numerical information would need a family such as `LinearSoftmax`, because a table cannot list every possible observation.

The linear models encode categorical parents as one-hot features in their declared state order and use numerical values directly, flattening arrays. There is no automatic scaling or addition of interaction features. A custom family implements the {py:class}`PolicyFamily` interface, and a custom parameter model for `DistributionPolicy` implements {py:class}`ParameterModel`.

## How does solve() fit both policies?

{py:func}`solve` uses policy gradients. At each update, it simulates complete days under the current promotion and order rules. It adds revenue and the two negative costs, then uses those total utilities to estimate how changing the action probabilities or densities affects expected profit. Adam uses that estimate to update the families' parameters.

The promotion and order policies are fitted together because their effects depend on one another. An order that works well when the promotion runs may be too much stock when it is skipped. A promotion's extra demand only helps if the owner orders enough ice cream to serve it. Each fitted rule still receives only its declared information parents.

The returned policy is the final training iterate, evaluated on fresh scenarios. The optimizer does not select a best checkpoint or report convergence diagnostics, and it gives no global or local optimality guarantee, even within the declared families.

## When does solve() fail?

{py:func}`solve` checks the diagram and its fitting settings before training:

- **The diagram is incomplete or structurally invalid.** Missing parents, cycles and unconfigured nodes are reported by validation.
- **The diagram has no decisions.** There is nothing to optimise, so there is no policy to return.
- **A decision has no policy family.** Every decision must declare a family.
- **A family does not fit the declared action or information space.** For example, `TabularSoftmax` requires scalar categorical information and categorical actions. A plain Normal policy can leave the order's bounded domain, so it cannot be used there without truncation.
- **The fitting settings are invalid.** `num_samples` and `num_steps` must be positive, and `learning_rate` must be positive and finite.

These cases raise `ValueError`. Continuous actions must be scalar, with a fixed distribution support inside their domain. Unsupported distribution metadata may raise `NotImplementedError`. A chance distribution that conflicts with its declared states also raises `ValueError`.

If training produces non-finite parameters or the final utility estimate is non-finite, the solver raises {py:class}`SolveError`. A completed call with finite outputs still requires us to assess the fitted policies, as the two shop fits above illustrate.

## Next steps

- [Inference](inference.md) explores profit under the fitted policies, conditions on observations, and compares the shop's outcomes under a different stocking policy.
