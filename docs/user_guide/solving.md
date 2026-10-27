---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving a diagram

Once we are happy with the way our influence diagram models the decision problem, we can use {py:func}`solve` to fit a policy that aims to maximise expected profit.

This page shows how to read what it returns, how to assess the result, and how to check that the actions we offered are the right ones. It also explains how {py:func}`solve` finds the policy and when it fails.

We continue with the ice cream shop from the [Building a diagram](building_a_diagram.md) page:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    TabularSoftmax,
    UtilityNode,
    solve,
)

DEMAND_MEANS = jnp.array([40.0, 120.0])
BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1
BOOKINGS_BOOST = jnp.array([1.0, 1.15])
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
        name="order",
        parents=("weather",),
        states=tuple(str(n) for n in ORDERS),
        policy_family=TabularSoftmax(),
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

diagram
```

## Reading the solution

{py:func}`solve` returns a {py:class}`Solution` object, which we can read with {py:meth}`~Solution.render`:

```{code-cell} ipython3
solution = solve(diagram)
print(solution.render())
```

The policy tells the owner how many ice creams to order for each kind of weather, accounting for uncertainty in `busyness`, bookings and daily demand. It does so by assigning probabilitities to the available orders.

A {py:class}`Solution` has two attributes:

- `policy` has one fitted {py:class}`CategoricalPolicy` per decision. Each rule stores its family, parameters, information parents and action labels.
- `expected_utility` estimates the expected total utility when the shop follows this policy. Here, that is the shop's expected profit.

We can inspect a rule directly through `probabilities`. It returns the action probabilities in the decision's `states`. To compare the most likely stock levels and their probabilities for cloudy and sunny days:

```{code-cell} ipython3
weather = jnp.array([0, 1])
probabilities = jax.vmap(solution.policy["order"].probabilities)(weather=weather)
print("most likely stock:", ORDERS[probabilities.argmax(axis=1)])
print("probability:", probabilities.max(axis=1))
```

These are the most likely orders, but the fitted rule still samples from all its action probabilities. The reported expected profit describes that stochastic policy.

## How much can we trust the result?

Training depends on the simulated days and on how long we let it run. `num_samples` sets the number of scenarios per update and in the final utility estimate, while `num_steps` sets the number of updates. The defaults are 10,000 samples and 500 updates. `learning_rate` controls the Adam step size and defaults to 0.05.

We can try a shorter run and change the random key to see how the fitted result varies:

```{code-cell} ipython3
for steps, seed in ((100, 0), (500, 0), (500, 1)):
    result = solve(diagram, num_steps=steps, rng_key=jax.random.PRNGKey(seed))
    probabilities = jax.vmap(result.policy["order"].probabilities)(weather=weather)
    print(f"updates={steps}, seed={seed}")
    print("stock:", ORDERS[probabilities.argmax(axis=1)])
    print("probability:", probabilities.max(axis=1))
    print(f"expected profit: {result.expected_utility:.1f}")
```

All three runs favour orders of 80 and 200, but after 100 updates their probabilities are only about 55% and 68%. After 500 updates, both are above 98%.

```{admonition} Choosing a training budget
:class: info

More samples, more updates and different random keys help in different ways:

- Increase `num_samples` to reduce noise in each training gradient estimate. This also makes the final expected-utility estimate more precise.
- Increase `num_steps` to give the optimizer more opportunities to improve the policy. More updates do not guarantee a better result.
- Try several `rng_key` values to check how consistently training finds similar policies and obtain candidates to compare.

When actions have similar expected payoffs, variation between simulated scenarios can make their differences hard to learn. It can be useful to try several random keys alongside a larger training budget.

Compare the fitted action probabilities and evaluate candidate policies on fresh scenarios with {py:func}`infer`, as described in [Inference](inference.md). Use enough evaluation draws to distinguish differences that matter for the decision. More evaluation draws improve our estimate of an existing policy's utility. Improving the policy itself requires training.
```

### What if a situation never appears?

`TabularSoftmax` has a separate row for each combination of its information parents' states. Rarely visited rows receive little training, so we should be cautious about their probabilities.

A combination might also be impossible. For example, a successful test result cannot follow a decision not to run the test. Such a row does not affect expected utility because it is never used. More samples cannot make an impossible combination appear.

## Are the actions the right ones?

The most likely order on a sunny day is 200 ice creams, which is the largest order on the menu. Now, when the policy favours the edge of the action space like this, a better option may lie beyond it.

To check, we widen the action space up to 400 and solve again. The `profit` function reads `ORDERS` when it runs, so redefining `ORDERS` and the decision's `states` is enough:

```{code-cell} ipython3
ORDERS = jnp.arange(0, 401, 20)
diagram["order"].states = tuple(str(n) for n in ORDERS)

solution = solve(diagram)
sunny = solution.policy["order"].probabilities(weather=jnp.array(1))
print("most likely sunny-day stock:", ORDERS[sunny.argmax()])
print("probability:", sunny.max())
print(f"expected profit: {solution.expected_utility:.1f}")
```

The fitted rule now puts the most probability on ordering 260 ice creams on sunny days. The wider menu lets the shop consider orders above 200, which the original action space excluded. This recommendation still depends on the fitted policy's quality.

The action space also constrains us in the other direction. The solver only compares the options we list, so with a step of 20 it could never find an optimum of, let's say, 50.

## How does solve() find the policy?

{py:func}`solve` uses policy gradients. At each update, it simulates complete scenarios under the current rules and uses their total utilities to estimate how changing the action probabilities affects expected utility. Adam uses that estimate to update the families' parameters.

The returned policy is the final training iterate, evaluated on fresh scenarios. The optimizer does not select a best checkpoint or report convergence diagnostics, and it gives no global or local optimality guarantee.

### Where should Mia and Leo set up?

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
carts.add_node(
    DecisionNode("mia_spot", states=("beach", "park"), policy_family=TabularSoftmax())
)
carts.add_node(
    DecisionNode("leo_spot", states=("beach", "park"), policy_family=TabularSoftmax())
)
carts.add_node(
    UtilityNode(name="profit", parents=("mia_spot", "leo_spot"), values=carts_profit)
)
carts
```

Leo's best spot depends on where Mia decides to go, and the other way around: if Mia takes the beach, Leo does better at the park, and if Mia takes the park, Leo does better at the beach. But neither sees the other's choice, so their policies have to work without that information.

`solve` fits their two policies together, using the profit from both carts:

```{code-cell} ipython3
cart_solution = solve(carts)
print(cart_solution.render())
```

The fitted rules favour Mia taking the beach and Leo taking the park. That lets them reach both groups of customers and saves Leo the \$20 van fare. Their choices are still made separately: neither needs to see where the other went.

## Fitting a policy with continuous information

A decision with a continuous parent still chooses among its finite actions, but a table cannot list every possible observation. We choose a policy family when defining the decision, and the solver fits its parameters.

We use {py:class}`LinearSoftmax`: linear scores followed by softmax probabilities over the actions. Numerical parents supply raw values, with arrays flattened, while categorical parents supply one-hot features in their declared state order. There is no automatic scaling. This family is useful when a linear boundary can describe the action choice.

For a small example, suppose we observe a signal uniformly distributed between −1 and 1. Taking action `on` earns the signal's value, while `off` earns zero. We declare the family on the decision:

```{code-cell} ipython3
from pylimid import LinearSoftmax

signal_diagram = InfluenceDiagram()
signal_diagram.add_node(
    ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0))
)
signal_diagram.add_node(
    DecisionNode(
        "action",
        parents=("signal",),
        states=("off", "on"),
        policy_family=LinearSoftmax(),
    )
)
signal_diagram.add_node(
    UtilityNode(
        "payoff",
        parents=("signal", "action"),
        values=lambda signal, action: signal * action,
    )
)
fit = solve(
    signal_diagram,
    num_steps=300,
    num_samples=2_000,
    learning_rate=0.05,
    rng_key=jax.random.PRNGKey(0),
)
print(fit.render())
rule = fit.policy["action"]
print("Action probabilities at signal=0.8:", rule.probabilities(signal=jnp.array(0.8)))
```

The returned {py:class}`CategoricalPolicy` keeps the family, fitted parameters, parent bindings and action labels. It remains stochastic: `probabilities` returns probabilities in `states` order, and `sample` takes an explicit random key. Replacing it with the largest-probability action would define a different policy.

`TabularSoftmax` requires scalar categorical information, while `LinearSoftmax` accepts numerical and categorical parents together. A custom family implements the {py:class}`PolicyFamily` interface. Choosing a suitable family is part of defining the decision problem.

## When does solve() fail?

{py:func}`solve` checks the diagram and its fitting settings before training:

- **The diagram has no decisions.** In that case there is nothing to optimise, so there is no policy to return.
- **A decision has no policy family.** Every decision must declare a family.
- **A tabular family receives numerical or array-valued information.** `TabularSoftmax` requires scalar parent-state indices and named finite states.
- **The fitting settings are invalid.** `num_samples` and `num_steps` must be positive, and `learning_rate` must be positive and finite.

These cases raise `ValueError`. A chance distribution that conflicts with its declared states also raises `ValueError`. If training produces non-finite parameters or the final utility estimate is non-finite, the solver raises {py:class}`SolveError`.
