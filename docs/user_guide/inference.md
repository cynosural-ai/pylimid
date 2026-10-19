---
file_format: mystnb
kernelspec:
  name: python3
---

<!--
Possible future additions to this guide:

- Updating beliefs about an unknown cause from an observed outcome. For example, given sunny weather and unusually high observed demand, how likely are high hotel bookings? This would complement the current examples, which condition on known inputs and explore downstream outcomes.
- Querying a discrete variable and reading its posterior probabilities with Posterior.marginal(), paired with the state's labels. The bookings question above could introduce this without adding another model.
- Interpreting continuous summaries such as Posterior.std() and Posterior.hdi(). Explain that an interval describes the variability of the queried quantity, rather than the sampling error in its estimated mean, and distinguish outcome uncertainty from Monte Carlo estimation uncertainty.
- Briefly explaining how infer() obtains its draws: forward sampling when no chance node is observed; discrete enumeration when evidence is present and all relevant latent variables are discrete; NUTS when evidence is present and a relevant latent variable is continuous. Explain num_samples and rng_key alongside the examples, and link to NumPyro for MCMC background. The current infer() result does not expose sampler diagnostics.

A small scalar-prior example could also show how an observation updates an uncertain model parameter. A full learning-from-data guide can wait for supported workflows for repeated observations and carrying posterior uncertainty into solve().
-->

# Inference

Finding the optimal solution to a decision problem is great, but PyLIMID's capabilities do not stop there. Following the ice cream shop example, the owner may have questions beyond how much to order. For instance:

- What is the distribution of profits when following the recommended policy?
- What happens if the owner follows a different policy?
- Could the shop run out of stock on a sunny day?

{py:func}`infer` lets us explore these and other questions by taking any observations we have into account and then drawing values from the model.

We use [the same influence diagram as before](solving.md), with an order menu up to 400:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, infer, solve

DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])
ORDERS = jnp.arange(0, 401, 20)


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
    ChanceNode(
        name="demand",
        parents=("weather", "busyness", "bookings"),
        dist=lambda weather, busyness, bookings: dist.Gamma(
            4.0, 4.0 / (DEMAND_MEANS[weather] * busyness * BOOKINGS_BOOST[bookings])
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

## What is the distribution of profits when following the recommended policy?

To see the range and shape of the outcomes under the recommended policy, we first solve the diagram:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

And then we ask {py:func}`infer` for the distribution of the `profit` node, passing the policy from the solution:

```{code-cell} ipython3
recommended_profit = infer(
    diagram,
    query=["profit"],
    policy=solution.policy,
    num_samples=50_000,
)["profit"]
round(recommended_profit.mean().item(), 1)
```

The result is a {py:class}`Posterior` object whose `values` are a JAX array containing the profit draws. Each draw represents the profit on one possible day (i.e., the model draws values for weather, `busyness`, hotel bookings and demand, and the policy chooses the corresponding order value).

Numerical summaries such as `mean()` also return JAX arrays. For this scalar profit node, the mean is a zero-dimensional array; `.item()` converts it to a Python number for display above.

```{admonition} Why the estimates differ
:class: info

Here {py:func}`solve` estimates an expected profit of \$632, while {py:func}`infer` estimates \$634, both under the same policy. This small mismatch comes from using different simulated days.
```

If we plot the resulting draws, we get the following long-tailed distribution:

```{figure} /_static/figures/inference/recommended_policy_profit.png
:alt: Histogram of daily profit under the recommended policy, with a line marking its mean.
:width: 90%
```

The spikes at \$720 and \$2,340 correspond to selling all 80 ice creams on a cloudy day or all 260 ice creams on a sunny day. However, once the shop sells out, higher demand cannot increase that day's profit.

### What if today is sunny and hotel bookings are high?

The distribution above includes all kinds of days because we have not provided any observations. Suppose today is sunny and the owner obtains a report showing that bookings at nearby hotels are high. For simplicity, we assume that the report is free and accurately identifies the booking state.

Current `solution.policy` does not take into account that the owner has information about hotel bookings. To represent this, we add `bookings` as a parent of `order`:

```{code-cell} ipython3
from copy import deepcopy

report_diagram = deepcopy(diagram)
report_diagram.add_arc("bookings", "order")

report_diagram
```

If we run {py:func}`solve` again, we can see the new policy recommends an order for each combination of weather and hotel bookings:

```{code-cell} ipython3
report_solution = solve(report_diagram, num_samples=50_000)
print(report_solution.render(report_diagram))
```

We can now use {py:func}`infer` to see the distribution of profits for a sunny day with high bookings under that policy. The `observed` argument records these observations using state indices: `1` for sunny weather and `1` for high bookings. Busyness and demand remain unobserved:

```{code-cell} ipython3
sunny_profit = infer(
    report_diagram,
    query=["profit"],
    policy=report_solution.policy,
    observed={"weather": 1, "bookings": 1},
    num_samples=50_000,
)["profit"]
round(sunny_profit.mean().item(), 1)
```

The new policy recommends ordering 280 ice creams, with an estimated average profit of about \$1,310. We can also see the distribution of profits for that day:

```{figure} /_static/figures/inference/sunny_policy_profit.png
:alt: Histogram of profit on a sunny day with high hotel bookings under the policy that uses both variables, with a line marking its mean.
:width: 90%
```

## What happens if the owner follows a different policy?

We return to the original diagram. Suppose the owner wants to know what would happen if the shop ordered 160 ice creams every day, regardless of the weather. We can ask {py:func}`infer` the same question about the `profit` node, but pass this fixed order (which corresponds to index `8`) as the policy:

```{code-cell} ipython3
alternative_profit = infer(
    diagram,
    query=["profit"],
    policy={"order": 8},
    num_samples=50_000,
)["profit"]
round(alternative_profit.mean().item(), 1)
```

The mean of these draws gives an estimated expected profit of about \$562, compared with \$634 under the recommended policy. We can also compare both distributions, using no observations in either case:

```{figure} /_static/figures/inference/policy_comparison.png
:alt: Histograms comparing profit under the recommended weather-based policy and a fixed order of 160 ice creams, with lines marking their means.
:width: 90%
```

This policy orders more than the recommended 80 ice creams on cloudy days and fewer than the recommended 260 on sunny days. Its spike at \$1,440 corresponds to selling all 160 ice creams, which is the highest profit it can make in a day.

## Could the shop run out of stock on a sunny day?

So far, we have used {py:func}`infer` to explore profit under different policies, but we can also ask questions about other variables in the model. Suppose today is sunny and only 150 ice creams arrive from the supplier. How likely is it that demand exceeds the available stock?

To answer this question, we ask for the distribution of `demand` given sunny weather. In this example, the owner has no hotel report, so bookings remain unknown:

```{code-cell} ipython3
sunny_demand = infer(
    diagram,
    query=["demand"],
    observed={"weather": 1},
    num_samples=50_000,
)["demand"]
round(sunny_demand.mean().item(), 1)
```

We do not pass a policy because, in this model, demand is not affected by what the owner orders. The estimated average demand is about 150 ice creams. Below we show the posterior distribution of `demand` given sunny weather:

```{figure} /_static/figures/inference/sunny_demand.png
:alt: Histogram of demand given sunny weather, with a line marking the available stock of 150 ice creams.
:width: 90%
```

To estimate the probability of running out, we count how many demand draws exceed 150 and divide by the total:

```{code-cell} ipython3
stockout_probability = (sunny_demand.values > 150).mean()
f"{stockout_probability:.1%}"
```

## Learning from observed counts

Suppose we model arrivals in another shop with a {py:class}`Poisson <numpyro.distributions.discrete.Poisson>` distribution and an unknown arrival rate. We place a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` prior on that rate. After observing four arrivals in one time period, we can update the rate:

```{code-cell} ipython3
arrival_model = InfluenceDiagram()
arrival_model.add_node(ChanceNode(name="rate", dist=lambda: dist.Gamma(1.0, 1.0)))
arrival_model.add_node(
    ChanceNode(name="arrivals", parents=("rate",), dist=lambda rate: dist.Poisson(rate))
)
arrival_rate = infer(arrival_model, ["rate"], observed={"arrivals": 4})["rate"]
arrival_rate.mean()
```

The observation is the actual count, without state labels. Because `arrivals` is known, inference evaluates its likelihood while sampling the continuous rate. Observed counts can also inform categorical parents.

With no evidence, we can forward sample numerical counts and any utilities that depend on them. With evidence, every numerical count remaining unknown in the relevant model currently causes an {py:class}`InferenceError`, including finite Binomial counts. This also applies when the count is only an ancestor of a query or observation. For example, observing a noisy report of arrivals still leaves the arrival count unknown; querying only the rate does not remove that unknown count.
