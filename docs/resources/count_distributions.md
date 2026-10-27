---
file_format: mystnb
kernelspec:
  name: python3
---

# Count distributions

Counts represent whole-number quantities, such as customer arrivals or defective products. We define them with a numerical distribution and leave `states` unset. Their values are actual counts, so downstream distributions and utilities receive those numbers directly.

This page uses standalone examples to explain forward sampling, decisions involving counts, and the cases supported by posterior inference.

## Defining and sampling a count

Suppose a service counter receives an average of twelve customers per time period. We model the number of arrivals with a {py:class}`Poisson <numpyro.distributions.discrete.Poisson>` distribution and use {py:func}`infer` to simulate possible counts:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer

arrivals = InfluenceDiagram()
arrivals.add_node(ChanceNode(name="customers", dist=lambda: dist.Poisson(12.0)))
customers = infer(arrivals, ["customers"])["customers"]
customers.mean()
```

Other numerical count distributions include {py:func}`Binomial <numpyro.distributions.discrete.Binomial>` and {py:class}`NegativeBinomial2 <numpyro.distributions.conjugate.NegativeBinomial2>`. Counts do not need a label for each possible value, even when their range is finite.

The returned {py:class}`Posterior` supports `mean()`, `std()` and `hdi()`. We can also estimate an event probability directly from its draws:

```{code-cell} ipython3
jnp.mean(customers.values >= 20)
```

`marginal()` requires named categories and does not apply to numerical counts. Count intervals can include unsampled values and are not necessarily smallest credible sets. These summaries are Monte Carlo estimates.

## Using counts in a decision problem

Suppose we must choose capacity before customers arrive. Each customer served earns five units, and each unit of capacity costs one. We add a {py:class}`DecisionNode` with two possible capacities and a {py:class}`UtilityNode` that uses the customer count. We choose {py:class}`TabularSoftmax` to fit the action probabilities:

```{code-cell} ipython3
from pylimid import DecisionNode, TabularSoftmax, UtilityNode, solve

CAPACITIES = jnp.array([10, 20])
arrivals.add_node(
    DecisionNode(
        name="capacity", states=("10", "20"), policy_family=TabularSoftmax()
    )
)
arrivals.add_node(
    UtilityNode(
        name="profit",
        parents=("capacity", "customers"),
        values=lambda capacity, customers: (
            5.0 * jnp.minimum(CAPACITIES[capacity], customers) - CAPACITIES[capacity]
        ),
    )
)
solution = solve(arrivals)
print(solution.render())
```

The fitted rule favours capacity 20. {py:func}`solve` fits the policy using simulated customer counts. The count is unknown when we choose capacity, so it is a parent of `profit` but not of `capacity`. The rule uses the same action probabilities for every time period. Unbounded counts such as Poisson work in this forward simulation.

### Choosing after observing the count

Suppose we can instead choose capacity after counting the customers waiting to be served. We add `customers` as a parent of `capacity` so the rule can use that information.

`TabularSoftmax` requires named categorical information, so it cannot directly use this numerical count. We choose {py:class}`LinearSoftmax`, which accepts the actual count as an input, and fit a new policy:

```{code-cell} ipython3
from copy import deepcopy

from pylimid import LinearSoftmax

informed_arrivals = deepcopy(arrivals)
informed_arrivals.add_arc("customers", "capacity")
informed_arrivals["capacity"].policy_family = LinearSoftmax()
informed_solution = solve(informed_arrivals)
rule = informed_solution.policy["capacity"]
for customers_waiting in (8, 16):
    print(
        f"{customers_waiting} customers:",
        rule.probabilities(customers=jnp.array(customers_waiting)),
    )
```

The probabilities are in the decision's state order, `("10", "20")`. During training, the rule receives the customer count sampled in each scenario. For a specific queue, we can then pass the known count through `observed` when evaluating its profit:

```{code-cell} ipython3
known_profit = infer(
    informed_arrivals,
    ["profit"],
    observed={"customers": 16},
    policy=informed_solution.policy,
)["profit"]
round(known_profit.mean().item(), 1)
```

The count is observed in this query, so its unbounded range does not prevent conditioning on it. An unknown Poisson count needed by a posterior query remains unsupported, as explained under [Current limits](#current-limits). See [Solving](../user_guide/solving.md#fitting-a-policy-with-continuous-information) for more about policy families and numerical information.

## Learning from observed counts

Suppose the arrival rate is unknown. We place a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` prior on it, then observe four arrivals in one time period. We use {py:func}`infer` to update the rate:

```{code-cell} ipython3
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer

arrival_model = InfluenceDiagram()
arrival_model.add_node(ChanceNode(name="rate", dist=lambda: dist.Gamma(1.0, 1.0)))
arrival_model.add_node(
    ChanceNode(name="arrivals", parents=("rate",), dist=lambda rate: dist.Poisson(rate))
)
arrival_rate = infer(arrival_model, ["rate"], observed={"arrivals": 4})["rate"]
arrival_rate.mean()
```

The observation is the actual count, without state labels. Because `arrivals` is known, inference evaluates its likelihood while sampling the continuous rate. Observed counts can also inform categorical parents. The count's unbounded range is not a problem here because inference does not need to explore unknown arrival counts.

## Inferring an unknown count from an alarm

Suppose a batch contains ten products. We model the number of defects with a {py:func}`Binomial <numpyro.distributions.discrete.Binomial>` distribution. An alarm is more likely when there are at least three defects, but it is not perfectly reliable. After the alarm goes off, we use {py:func}`infer` to estimate the defect count:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer

inspection = InfluenceDiagram()
inspection.add_node(
    ChanceNode(name="defects", dist=lambda: dist.Binomial(total_count=10, probs=0.2))
)
inspection.add_node(
    ChanceNode(
        name="alarm",
        parents=("defects",),
        states=("off", "on"),
        dist=lambda defects: dist.Bernoulli(jnp.where(defects >= 3, 0.9, 0.1)),
    )
)
defects = infer(inspection, ["defects"], observed={"alarm": 1})["defects"]
defects.mean()
```

Here the range is fixed: zero through ten. NumPyro sums over those possibilities and returns posterior draws containing actual defect counts. The returned summaries remain Monte Carlo estimates. A success probability that depends on a categorical or continuous parent is supported as long as the number of trials stays fixed.

## Current limits

When conditioning on chance-node evidence, an unknown count must have a scalar distribution and a fixed integer range starting at zero that NumPyro can enumerate. Examples include Binomial and Beta-Binomial. For Binomial, use a Python integer `total_count`. Bounds computed with JAX inside the distribution callable may become non-static. Zero-trial counts are clamped to zero exactly.

The following cases remain unsupported:

- **Parent-dependent bounds.** For example, a Binomial defect count whose number of trials depends on an unknown batch size cannot currently be inferred. It can still be sampled forward or supplied as an observation.
- **Unknown unbounded counts.** For example, observing a noisy report of Poisson arrivals leaves the arrival count unknown. Posterior inference cannot currently explore that count's unbounded range. This also applies when we query only its rate, because an unknown count needed to connect the rate to the observation still requires inference. Unbounded counts needed by a posterior query cause an {py:class}`InferenceError`. Support is never silently truncated.
- **Shifted ranges and batched count latents.** Unknown counts with ranges starting above zero, or multiple counts returned together as a batch, are not supported by posterior inference. Batched scalar-event counts retain component-wise numerical summaries in forward sampling and can be observed as likelihoods.
- **Vector events.** Distributions such as Multinomial, where a single draw is a vector of related counts, are outside the current support.

NumPyro and JAX report non-static or heterogeneous enumeration bounds directly. See [Inference](../user_guide/inference.md) for querying other variables and evaluating a policy's outcomes.
