---
file_format: mystnb
kernelspec:
  name: python3
---

# Inference

Finding the optimal solution to a decision problem is great, but PyLIMID's capabilities do not stop there. Following the ice cream shop example, the owner may have questions beyond how much to order. For instance:

- What is the distribution of profits when following the recommended policy?
- What is the probability of losing money with this policy?
- How would those outcomes change under a different policy?

{py:func}`infer` lets us explore these and other questions by taking any observations we have into account and then drawing values from the model.

We use the same influence diagram as before. The code below rebuilds it so that the examples can be run on their own, using the order menu up to 400 from the [Solving a diagram](solving.md) page:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, infer, solve

DEMAND_MEANS = jnp.array([40.0, 120.0])
BOOKINGS_BOOST = jnp.array([1.0, 1.5])
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

## What is the distribution of profits when following the recommended policy?

To see the range and shape of the outcomes we get from following the recommended policy, we solve the diagram as we did before:

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
round(recommended_profit.mean(), 1)
```

The result is a {py:class}`Posterior` object whose `values` contain the profit draws. Each draw represents the profit on one possible day (i.e., the model draws values for weather, hotel bookings and demand, and the policy chooses the corresponding order value).

```{admonition} Why the estimates differ
:class: info

Here {py:func}`solve` estimates an expected profit of \$569, while {py:func}`infer` estimates \$571, both under the same policy. This small mismatch comes from using different simulated days.
```

If we plot the resulting draws, we get the following long-tailed distribution:

```{figure} /_static/figures/inference/recommended_policy_profit.png
:alt: Histogram of daily profit under the recommended policy, with a line marking its mean.
:width: 90%
```

The spikes at \$720 and \$2,160 correspond to selling all 80 ice creams on a cloudy day or all 240 ice creams on a sunny day. However, once the shop sells out, higher demand cannot increase that day's profit.

### What if today is sunny and hotel bookings are high?

The distribution above includes all kinds of days because we have not provided any observations. Suppose we know that today is sunny and the owner obtains a report showing that bookings at nearby hotels are high. For simplicity, we assume that the report is free and accurately identifies the booking state.

The current policy takes weather into account, but not hotel bookings. Since both are now known before placing the order, we add `bookings` as a parent of `order` and then run {py:func}`solve` again:

```{code-cell} ipython3
from copy import deepcopy

report_diagram = deepcopy(diagram)
report_diagram.add_arc("bookings", "order")
report_solution = solve(report_diagram, num_samples=50_000)
print(report_solution.render(report_diagram))

report_diagram
```

The new policy recommends an order for each combination of weather and hotel bookings. We can now use {py:func}`infer` to see the distribution of profits for a sunny day with high bookings under that policy. Demand remains unobserved:

```{code-cell} ipython3
sunny_profit = infer(
    report_diagram,
    query=["profit"],
    policy=report_solution.policy,
    observed={"weather": 1, "bookings": 1},
    num_samples=50_000,
)["profit"]
round(sunny_profit.mean(), 1)
```

For a sunny day with high bookings, the new policy recommends ordering 300 ice creams, with an estimated average profit of about \$1,436. We can also see the distribution of profits for that day:

```{figure} /_static/figures/inference/sunny_policy_profit.png
:alt: Histogram of profit on a sunny day with high hotel bookings under the policy that uses both variables, with a line marking its mean.
:width: 90%
```

## What is the probability of losing money?

> **Draft note:** Ask a specific question about the recommended policy's distribution from the previous section. Reuse its profit draws to estimate loss probability by counting negative profits and dividing by the total. No new simulation is needed for this calculation. This answers the second opening question and shows how the distribution gives us information beyond average profit. Keep the calculation available for comparing policies in the next section and for asking about demand later.

## How would those outcomes change under a different policy?

> **Draft note:** Start with a fixed order, then optionally a custom policy that responds to weather. Pass each to `infer` and obtain its profit distribution and estimated expected utility with the same diagram. Explain that fixed actions are indices into `ORDERS`, and that a contingent table covers every combination of the decision's parent states in parent order. Show the distributions on the same axes and compare their means and spread. Reuse the previous section's calculation to compare loss probabilities too. Introduce `std()` or an outcome interval only if they help describe that comparison. The mean of the alternative's distribution is its expected utility; the maximum expected utility refers to the best policy. This answers the third opening question and lets readers assess how another choice changes both average profit and risk.

## Could the shop run out of stock on a sunny day?

> **Draft note:** This is the final section of the page. Make the transition explicit: finding and evaluating a policy are useful, and `infer` also lets us explore the distributions of other variables in the model. Suppose only 150 ice creams arrive from the supplier, fewer than the owner ordered, and the day is sunny. Query `demand` with the observed sunny weather, reusing the observation syntax and state indices introduced in the first section. Show the conditional demand distribution, then estimate the probability of demand exceeding the 150 available. In this model, available stock does not change customer demand, so this question uses the existing diagram and needs no order policy. It assesses the risk of running out; calculating profit after a delivery shortfall would also require representing the actual available stock in the profit calculation. A cloudy-day comparison can follow if useful. The earlier verified demand example gave about 35.5% on sunny days and 0.3% on cloudy days with 50,000 draws; rerun it when drafting the final text. Close by connecting this example to asking questions about other model variables. Keep parameter learning on a future page.
