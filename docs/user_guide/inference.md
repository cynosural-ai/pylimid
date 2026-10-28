---
file_format: mystnb
kernelspec:
  name: python3
---

# Inference

Finding a good policy for a decision problem is great, but PyLIMID's capabilities do not stop there. Following the ice cream shop example, the owner may have questions beyond whether to promote and how much to order. For instance:

- What is the distribution of profits when following the fitted policies?
- What happens on sunny days with high hotel bookings?
- What happens if the owner follows a different stocking policy?
- Could the shop run out of stock on a sunny day?

{py:func}`infer` lets us explore these and other questions by taking any observations we have into account and then drawing values from the model.

We use [the same diagram as before](solving.md), with the wider order family from the end of that page. The underlying Normal's scale is fixed at 1 kilogram, and the order is still bounded between zero and the freezer's capacity. We keep the three utility nodes for revenue, stock cost and promotion cost:

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
    infer,
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
UTILITIES = ["revenue", "stock_cost", "promotion_cost"]


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
            fixed_parameters={"low": 0.0, "high": CAPACITY, "scale": 1.0},
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

## What is the distribution of profits under the fitted policies?

We first fit the two policies with the same settings as in the Solving guide:

```{code-cell} ipython3
solution = solve(diagram, num_steps=2_000)
print(solution.render())
```

We then ask {py:func}`infer` for draws from the three utility nodes, passing the fitted policies from the solution:

```{code-cell} ipython3
outcomes = infer(
    diagram,
    query=UTILITIES,
    policy=solution.policy,
    num_samples=50_000,
    rng_key=jax.random.PRNGKey(10),
)
profit_draws = (
    outcomes["revenue"].values
    + outcomes["stock_cost"].values
    + outcomes["promotion_cost"].values
)
print(f"Mean daily profit: ${profit_draws.mean():.1f}")
print(f"Standard deviation across days: ${profit_draws.std():.1f}")
```

The result has one {py:class}`Posterior` object for each name in `query`. Its `values` are a JAX array of draws for that variable. Values at the same position in these three arrays describe the same day, so adding them gives that day's total profit, including both costs.

Each day includes draws for weather, `busyness`, hotel bookings and demand, together with the actions sampled by the fitted promotion and order rules. Keeping related quantities in the same call preserves their relationships. Arrays from separate inference calls should not be treated as matching days.

The estimated mean profit is about \$693.7 per day, with a standard deviation across days of about \$630.9. The average alone does not tell the owner how much the shop will earn on a particular day:

```{figure} /_static/figures/inference/fitted_policy_profit.png
:alt: Histogram of total daily profit under the fitted promotion and weather-dependent order policies, with a line marking the mean
:width: 90%
```

The distribution includes both quiet and busy days. When the shop sells all its stock, extra demand cannot increase that day's profit. Since the order varies continuously, profit on those sold-out days also varies.

### Why do solve() and infer() give different averages?

The two calls use different simulated days. {py:func}`solve` estimates expected profit on fresh days after training, while the call to {py:func}`infer` above evaluates the same fitted policies again. It keeps their learned parameters fixed.

Here, with no observations, `infer` uses independent forward draws. We can estimate the standard error of the mean profit from their spread and number:

```{code-cell} ipython3
mean_error = profit_draws.std() / jnp.sqrt(profit_draws.size)
print(f"Estimated standard error of the mean: ${mean_error:.1f}")
```

`solve` estimates about \$689.2 per day, while this call to `infer` gives about \$693.7 under the same policies. The standard error for the latter estimate is about \$2.8, much smaller than the variation in profit across individual days.

`num_samples` sets the number of draws in an inference call. Increasing it makes the evaluation more precise. The `rng_key` controls randomness and defaults to a fixed key, so repeated calls with the same model and settings are reproducible. We use an explicit key here to keep this evaluation separate from training.

## What if today is sunny and hotel bookings are high?

The distribution above includes all kinds of days because we have not provided any observations. Suppose today is sunny and bookings at nearby hotels are high. We can condition the model on those facts, using state index `1` for both `sunny` and `high`:

```{code-cell} ipython3
sunny_outcomes = infer(
    diagram,
    query=UTILITIES,
    observed={"weather": 1, "bookings": 1},
    policy=solution.policy,
    num_samples=10_000,
    rng_key=jax.random.PRNGKey(10),
)
sunny_profit = (
    sunny_outcomes["revenue"].values
    + sunny_outcomes["stock_cost"].values
    + sunny_outcomes["promotion_cost"].values
)
print(f"Mean profit on a sunny day with high bookings: ${sunny_profit.mean():.1f}")
```

These draws describe days consistent with both observations. The fitted rules still use only their declared information parents: the order can respond to weather and promotion, but it does not receive hotel bookings. Conditioning a query does not give a decision a new information parent or refit its policy.

### What if the owner can use a hotel report?

Suppose the owner can obtain a free report that accurately identifies hotel bookings before ordering. To represent this, we add `bookings` as a parent of `order` and fit the policies again. The report arrives in the morning, so the promotion is still chosen without it:

```{code-cell} ipython3
diagram.add_arc("bookings", "order")
report_solution = solve(diagram, num_steps=2_000)
print(report_solution.render())

report_outcomes = infer(
    diagram,
    query=UTILITIES,
    observed={"weather": 1, "bookings": 1},
    policy=report_solution.policy,
    num_samples=10_000,
    rng_key=jax.random.PRNGKey(10),
)
report_profit = (
    report_outcomes["revenue"].values
    + report_outcomes["stock_cost"].values
    + report_outcomes["promotion_cost"].values
)
print(f"Mean profit with the hotel report: ${report_profit.mean():.1f}")
```

For sunny days with high bookings, the estimated mean profit is about \$1,500.7 under the original rules and \$1,512.9 under the rules fitted with access to the report. Both estimates depend on training and posterior sampling:

```{figure} /_static/figures/inference/hotel_report_comparison.png
:alt: Profit distributions on sunny days with high hotel bookings under the original policy and a newly fitted policy that can use a hotel report
:width: 90%
```

These numbers describe one kind of day under two fitted policies. Assessing the value of the report would require comparing well-trained policies across all days, together with any cost of obtaining it.

We return to the original information set before exploring other questions. Its fitted rules are still in `solution.policy`:

```{code-cell} ipython3
diagram.remove_arc("bookings", "order")
diagram["order"].parents
```

## What happens if the owner follows a different policy?

Suppose the owner orders 16 kilograms every day, regardless of the weather. We keep the fitted promotion rule and replace only the order rule with this fixed quantity:

```{code-cell} ipython3
alternative_outcomes = infer(
    diagram,
    query=UTILITIES,
    policy={"promotion": solution.policy["promotion"], "order": 16.0},
    num_samples=50_000,
    rng_key=jax.random.PRNGKey(10),
)
alternative_profit = (
    alternative_outcomes["revenue"].values
    + alternative_outcomes["stock_cost"].values
    + alternative_outcomes["promotion_cost"].values
)
print(f"Mean profit when always ordering 16 kg: ${alternative_profit.mean():.1f}")
```

Here `16.0` is the amount ordered in kilograms, because `order` is continuous. Both decisions influence profit, so we supply a rule for each. This call evaluates the alternative policy without training it.

The fixed order gives an estimated expected profit of about \$601.0 per day, compared with \$693.7 under the fitted policies. We can compare both distributions without conditioning on any observations:

```{figure} /_static/figures/inference/policy_comparison.png
:alt: Histograms comparing total profit under the fitted policies and a fixed order of 16 kilograms, keeping the fitted promotion rule in both cases
:width: 90%
```

The fixed order carries more stock than the fitted rule usually chooses on cloudy days and less on sunny days. Its prominent peak comes from selling all 16 kilograms. The same promotion rule is used in both comparisons.

## Could the shop run out of stock on a sunny day?

So far, we have explored profit, but we can also ask questions about other variables. Suppose today is sunny, the promotion is already running, and only 20 kilograms arrive from the supplier. How likely is it that demand exceeds the available stock?

We ask for the distribution of `demand` given sunny weather, fixing the earlier promotion choice at `run`. There is no hotel report in this example, so bookings remain unknown:

```{code-cell} ipython3
sunny_demand = infer(
    diagram,
    query=["demand"],
    observed={"weather": 1},
    policy={"promotion": 1},
    num_samples=10_000,
    rng_key=jax.random.PRNGKey(10),
)["demand"]
print(f"Mean demand: {sunny_demand.mean().item():.1f} kg")
print(f"Standard deviation: {sunny_demand.std().item():.1f} kg")
lower, upper = sunny_demand.hdi()
print(f"95% highest-density interval: {lower:.1f} to {upper:.1f} kg")
```

The promotion affects demand, so we must supply its chosen action. The order does not affect demand and is not needed for this query. Decisions are bound through `policy`, while `observed` records chance-node observations.

The {py:class}`Posterior` summaries `mean()`, `std()` and `hdi()` return JAX arrays. For scalar quantities, `.item()` converts them to Python numbers for display. The highest-density interval is the narrowest interval containing the requested fraction of draws, 95% by default. It describes variation in demand across days consistent with our information.

```{figure} /_static/figures/inference/sunny_demand.png
:alt: Demand in kilograms on sunny days when the promotion runs, with a line marking 20 kilograms of available stock
:width: 90%
```

To estimate the probability of running out, we count the fraction of demand draws above 20 kilograms:

```{code-cell} ipython3
stockout_probability = (sunny_demand.values > 20.0).mean()
print(f"Probability of running out: {stockout_probability:.1%}")
```

Under this model, demand exceeds 20 kilograms on about 35% of sunny days when the promotion runs, even though average demand is about 18 kilograms. The 95% highest-density interval runs from about 2.9 to 36.2 kilograms, showing how much demand can vary.

## Can a busy day tell us anything about hotel bookings?

We can also reason in the other direction. Suppose the owner records 30 kilograms of customer demand on a sunny day when the promotion runs, including requests the shop could not fulfil. Does that make high hotel bookings more likely?

```{code-cell} ipython3
beliefs = infer(
    diagram,
    query=["bookings", "busyness"],
    observed={"weather": 1, "demand": 30.0},
    policy={"promotion": 1},
    num_samples=10_000,
    rng_key=jax.random.PRNGKey(10),
)
for label, probability in zip(
    diagram["bookings"].states, beliefs["bookings"].marginal(), strict=True
):
    print(f"{label} bookings: {probability:.1%}")
print(f"Posterior mean busyness: {beliefs['busyness'].mean().item():.2f}")
```

For a categorical variable, `marginal()` gives probabilities in the order of its state labels. The probability of high bookings rises from roughly two days out of seven in the prior to about 36.6%. The posterior mean of `busyness` rises to about 1.23. A busy day can reflect high hotel bookings, a busier season or ordinary daily variation, so both possible causes remain uncertain.

These are updated beliefs under the model. The inference call leaves the model's prior and the fitted policy parameters unchanged.

## How does infer() obtain the draws?

Without chance-node observations, {py:func}`infer` uses independent forward sampling, drawing possible days from the model and actions from the supplied rules. This is how we evaluated the fitted policies and the fixed 16-kilogram alternative.

With chance-node evidence, it conditions the model. Categorical variables and fitted categorical actions are enumerated. In our shop model, continuous variables remain unknown, so NumPyro's [NUTS sampler](https://num.pyro.ai/en/stable/mcmc.html#nuts) draws their posterior values, including a continuous action when it is relevant. For a fully discrete supported model, enumeration supplies the draws without NUTS. The returned summaries are Monte Carlo estimates in both cases.

Conditioned MCMC draws can be correlated, so the simple standard-error calculation for independent draws above does not apply directly. More draws and runs with different random keys can help us check whether the summaries are stable. The current result does not expose sampler diagnostics or provide a convergence guarantee. A wide demand or profit interval describes outcome uncertainty, while uncertainty in its estimated mean also depends on sampling reliability.

## Next steps

- [Common pitfalls](../resources/common_pitfalls.md) collects checks for model inputs, parent values and fitted policies.
- The [API reference](../api/pylimid.rst) describes the arguments and results of `infer` and `solve`.
