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

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, infer, solve

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

Suppose we must choose capacity before customers arrive. Each customer served earns five units, and each unit of capacity costs one. We add a {py:class}`DecisionNode` with two possible capacities and a {py:class}`UtilityNode` that uses the customer count:

```{code-cell} ipython3
CAPACITIES = jnp.array([10, 20])
arrivals.add_node(DecisionNode(name="capacity", states=("10", "20")))
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
print(solution.render(arrivals))
```

{py:func}`solve` simulates customer counts to compare expected profit for the two actions. The count is unknown when we choose capacity, so it is a parent of `profit` but not of `capacity`. Unbounded counts such as Poisson work in this forward simulation.

A numerical count cannot directly index a decision's policy table: information-set parents still need named finite states. For example, a policy that chooses capacity after observing the exact customer count is not currently supported. See [Solving](../user_guide/solving.md) for how information sets determine which inputs a policy can use.

## Learning from observed counts

Suppose the arrival rate is unknown. We place a {py:class}`Gamma <numpyro.distributions.continuous.Gamma>` prior on it, then observe four arrivals in one time period. We use {py:func}`infer` to update the rate:

```{code-cell} ipython3
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

Posterior inference supports unknown scalar counts with a fixed integer range starting at zero that NumPyro can enumerate, including Binomial and Beta-Binomial. For Binomial, use a Python integer `total_count`; bounds computed with JAX inside the distribution callable may become non-static. Zero-trial counts are clamped to zero exactly.

The following cases remain unsupported:

- **Parent-dependent bounds.** For example, a Binomial defect count whose number of trials depends on an unknown batch size cannot currently be inferred. It can still be sampled forward or supplied as an observation.
- **Unknown unbounded counts.** For example, observing a noisy report of Poisson arrivals leaves the arrival count unknown. Posterior inference cannot currently explore that count's unbounded range. This also applies when we query only its rate: an unknown count needed to connect the rate to the observation still requires inference. Unbounded counts needed by a posterior query cause an {py:class}`InferenceError`; support is never silently truncated.
- **Shifted ranges and batched count latents.** Unknown counts with ranges starting above zero, or multiple counts returned together as a batch, are not supported by posterior inference. Batched scalar-event counts retain component-wise numerical summaries in forward sampling and can be observed as likelihoods.
- **Vector events.** Distributions such as Multinomial, where a single draw is a vector of related counts, are outside the current support.

NumPyro and JAX report non-static or heterogeneous enumeration bounds directly. See [Inference](../user_guide/inference.md) for querying other variables and evaluating a policy's outcomes.
