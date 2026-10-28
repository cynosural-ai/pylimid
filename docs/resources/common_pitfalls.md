---
file_format: mystnb
kernelspec:
  name: python3
---

# Common pitfalls

A model can run successfully while answering a different question from the one we intended. This page shows a few checks for common modelling and analysis mistakes.

## Indexing a table with several parents

When a node has several discrete parents, chained indexing such as `table[parent_1][parent_2]` can give incorrect posterior results or errors during NumPyro's discrete enumeration. This behaviour is reported in [NumPyro issue #2252](https://github.com/pyro-ppl/numpyro/issues/2252).

To avoid it, we can use both indices in a single operation, `table[parent_1, parent_2]`. For example, if `demand` has discrete parents `weather` and `bookings`:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

DEMAND_MEANS = jnp.array([[40.0, 60.0], [120.0, 180.0]])


def demand_dist(weather, bookings):
    mean = DEMAND_MEANS[weather, bookings]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)
```

## Checking parent effects

Suppose we have two chance nodes, `weather` and `demand`, with `weather` as a parent of `demand`. The demand function uses the weather state's index (`0` for `"cloudy"`, `1` for `"sunny"`) to select a mean from a lookup table.

In the following example, we accidentally include only the cloudy-day mean in that table:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram

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
        name="demand",
        parents=("weather",),
        dist=lambda weather: dist.Gamma(
            concentration=4.0,
            rate=4.0 / jnp.array([40.0])[weather],
        ),
    )
)

print("validation:", diagram.validate())
for problem in diagram.probe_discrete_parents():
    print(problem.message)
```

{py:meth}`~InfluenceDiagram.validate` checks the declared parents against the function's arguments, but it does not evaluate their effects. [JAX clamps the out-of-range index](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing) to the cloudy-day value, so both weather states produce the same demand distribution.

{py:meth}`~InfluenceDiagram.probe_discrete_parents` detects that `weather` has no effect. The check runs only when called and reports a warning, since ignoring a parent can be intentional. It does not check whether the values themselves are correct.

## Using a state index as a quantity

Suppose an order decision has `states=("80", "160")`, representing two possible stock levels. Passing `policy={"order": 1}` selects the second action, which stocks 160 ice creams. The value passed to a child of `order` is still the index `1`. The labels do not automatically convert it into a quantity.

If we use `order` directly in a profit calculation, we calculate the profit from stocking one ice cream. Instead, we look up the stock level in an array that follows the decision's state order:

```{code-cell} ipython3
import jax.numpy as jnp

ORDERS = jnp.array([80, 160])


def profit(order, demand):
    stocked = ORDERS[order]
    return 10.0 * jnp.minimum(stocked, demand) - stocked


print("stock:", ORDERS[1].item())
print("profit:", profit(order=1, demand=120.0).item())
```

Categorical chance variables use the same index convention. Numerical counts, however, already contain the quantities they represent. See [Count distributions](count_distributions.md) for that distinction.

## Combining draws from separate calls

To calculate a difference, a ratio or a joint event involving several variables, we need values that belong to the same simulated scenario. Query those variables together with {py:func}`infer`.

For example, suppose a thermometer reading varies around the true temperature. We can estimate the spread of its measurement error by subtracting the temperature from the reading in each draw:

```{code-cell} ipython3
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer

readings = InfluenceDiagram()
readings.add_node(
    ChanceNode("temperature", dist=lambda: dist.Normal(20.0, 2.0))
)
readings.add_node(
    ChanceNode(
        "reading",
        parents=("temperature",),
        dist=lambda temperature: dist.Normal(temperature, 0.5),
    )
)

draws = infer(readings, ["temperature", "reading"], num_samples=5_000)
error = draws["reading"].values - draws["temperature"].values
round(error.std().item(), 2)
```

The result is close to the measurement standard deviation of 0.5 that we put in the model. Pairing unrelated temperature and reading draws would also include variation in the true temperature, changing the quantity we estimate.

`infer` guarantees that matching positions belong to the same draw within one call. We should not rely on that correspondence across separate calls, even if they use the same sample count or random key. The [Inference guide](../user_guide/inference.md#what-is-the-distribution-of-profits-under-the-fitted-policies) uses this correspondence to add revenue and cost draws.

## Replacing a fitted policy with its most likely actions

{py:func}`solve` returns fitted {py:class}`CategoricalPolicy` rules that sample from their action probabilities. Choosing only the action with the largest probability creates a different policy.

A small example makes the difference visible. Skipping earns nothing, while running earns two units. We use a short training run so both actions retain visible probability:

```{code-cell} ipython3
from pylimid import (
    DecisionNode,
    InfluenceDiagram,
    TabularSoftmax,
    UtilityNode,
    infer,
    solve,
)

choice = InfluenceDiagram()
choice.add_node(
    DecisionNode("action", states=("skip", "run"), policy_family=TabularSoftmax())
)
choice.add_node(
    UtilityNode("payoff", parents=("action",), values=lambda action: 2.0 * action)
)

solution = solve(choice, num_steps=20, num_samples=500)
rule = solution.policy["action"]
rule.probabilities()
```

We evaluate the fitted rule by passing `solution.policy` to `infer`. To compare it with always taking its most likely action, we pass that action's index as a fixed policy:

```{code-cell} ipython3
fitted_payoff = infer(
    choice, ["payoff"], policy=solution.policy, num_samples=5_000
)["payoff"]
most_likely = int(rule.probabilities().argmax())
fixed_payoff = infer(
    choice, ["payoff"], policy={"action": most_likely}, num_samples=5_000
)["payoff"]

print("fitted policy:", round(fitted_payoff.mean().item(), 2))
print("most likely action:", round(fixed_payoff.mean().item(), 2))
```

The fixed policy always earns two units. The fitted policy sometimes skips, so its expected payoff is lower. The difference comes from changing the policy we evaluate.

This decision has no information parents. For a decision with parents, converting a fitted rule to its most likely actions means selecting an action for each information assignment. See [Solving](../user_guide/solving.md#reading-the-solution) for reading fitted policies and [Inference](../user_guide/inference.md#what-happens-if-the-owner-follows-a-different-policy) for evaluating alternatives.
