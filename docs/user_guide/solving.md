---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving a diagram

Once we are happy with the way our influence diagram models the decision problem, we can use {py:func}`solve` to find the best policy.

This page shows how to read what it returns, how many samples we need to trust it, and how to check that the actions we offered are the right ones. It also explains how {py:func}`solve` finds the policy and when it fails.

We continue with the ice cream shop from the [Building a diagram](building_a_diagram.md) page:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

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
        name="order", parents=("weather",), states=tuple(str(n) for n in ORDERS)
    )
)
diagram.add_node(UtilityNode(name="profit", parents=("order", "demand"), values=profit))

diagram
```

## Reading the solution

{py:func}`solve` returns a {py:class}`Solution` object, which we can easily read with {py:meth}`~Solution.render`:

```{code-cell} ipython3
solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The policy tells the owner how many ice creams to order for each kind of weather, accounting for uncertainty in `busyness`, bookings and daily demand.

A {py:class}`Solution` has three attributes:

- `policy` has one entry per decision. An entry maps a situation to the recommended action.
- `expected_utility` estimates the expected total utility when the shop follows this policy. Here, that is the shop's expected profit.
- `method` records which solver ran (more on this at the end of the page).

`render` is just a friendly way of reading `policy`. Inside it, a situation is a tuple with the state index of each of the decision's parents, and the action is an index into the decision's `states`. Here `order` has a single parent, so `(0,)` means `"cloudy"` and `(1,)` means `"sunny"`. If we ever need to work with the policy directly, we can decode it ourselves with the nodes' `states`:

```{code-cell} ipython3
for (weather,), action in solution.policy["order"].items():
    print(f"{diagram['weather'].states[weather]}: stock {diagram['order'].states[action]}")
```

## How much can we trust the result?

PyLIMID uses sampling as its inference engine, so the result depends on how many random draws we take (`num_samples`) and the random key that generates them (`rng_key`). We can solve the diagram with a few different keys to see how much the answer depends on the draws:

```{code-cell} ipython3
for num_samples in (2_000, 50_000):
    for seed in range(4):
        result = solve(diagram, num_samples=num_samples, rng_key=jax.random.PRNGKey(seed))
        orders = [diagram["order"].states[a] for a in result.policy["order"].values()]
        print(f"{num_samples:>6} samples, seed {seed}: stock {orders}, profit {result.expected_utility:.1f}")
```

The policy is the same in every run, but the expected profit is not: with 2,000 samples it moves by about 18 between keys, and with 50,000 by about 5. So if we want a precise expected utility, we need more samples.

A good habit before relying on a result is to solve it several times with different `rng_key` values. If the policy changes, either two actions are nearly as good as each other or `num_samples` is too small to tell them apart.

On a small diagram, compiling the model can take much of the runtime, so increasing `num_samples` does not necessarily increase the runtime in proportion.

### What if a situation never appears?

Backward induction groups simulations by each combination of a decision's parent states. If a combination never appears, it cannot estimate which action is best for it. The solver completes the solve, assigns the first action in the decision's `states` to that combination, and warns with the decision name and affected parent states. Those policy entries have not been optimised.

A missing combination might be impossible. For example, if we decide not to run a test, a successful test result cannot follow. Its action will never be used. But a possible combination can also be absent simply because it is rare. The solver cannot distinguish those cases from the samples alone, so we should check the warning against the model and increase `num_samples` if the combination is possible. More samples cannot make an impossible combination appear.

## Are the actions the right ones?

The model recommends an order of 200 ice creams on a sunny day, which is the largest order on the menu. Now, when the best action sits at the edge of the action space like this, the true best may lie beyond it.

To check, we widen the action space up to 400 and solve again. The `profit` function reads `ORDERS` when it runs, so redefining `ORDERS` and the decision's `states` is enough:

```{code-cell} ipython3
ORDERS = jnp.arange(0, 401, 20)
diagram["order"].states = tuple(str(n) for n in ORDERS)

solution = solve(diagram, num_samples=50_000)
print(solution.render(diagram))
```

The best sunny-day order is now 260, so 200 was only the best option within the restricted action space we had defined. Unless the shop cannot stock more than 200, the owner should order 260 instead.

The action space also constrains us in the other direction. The solver only compares the options we list, so with a step of 20 it could never find an optimum of, let's say, 50.

## How does solve() find the policy?

So far we haven't had to think about how {py:func}`solve` finds the best policy, because the default, `method="auto"`, picks the solver method for us. It chooses between two table methods:

- `backward_induction` resolves one decision at a time, from the last to the first. It finds the best policy when the decisions can be resolved one after the other. PyLIMID checks this with {py:func}`~pylimid.solver.is_solvable` when we select `auto`. A diagram with a single decision, like ours, always qualifies.
- `scan` follows a brute-force approach. It simulates every possible policy in the action space and keeps the best one. It works on any diagram with categorical information sets but, since the number of policies grows exponentially with the number of situations a decision can face, it quickly becomes impractical. In our problem, `order` has 21 actions and 2 situations (`cloudy` and `sunny`), so there are $21^2 = 441$ policies, each simulated `num_samples` times. If the owner also had a report showing whether hotel bookings are normal or high for that day, there would be 4 situations and $21^4 = 194,481$ policies. If we add another two-state parent variable the scan is no longer an option, even though the diagram itself is still small.

### Example: when we can't use backward induction

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
carts.add_node(DecisionNode(name="mia_spot", states=("beach", "park")))
carts.add_node(DecisionNode(name="leo_spot", states=("beach", "park")))
carts.add_node(
    UtilityNode(name="profit", parents=("mia_spot", "leo_spot"), values=carts_profit)
)
carts
```

Leo's best spot depends on where Mia decides to go, and the other way around: if Mia takes the beach, Leo does better at the park, and if Mia takes the park, Leo does better at the beach. But neither sees the other's choice, so neither decision can be resolved first. As a result, backward induction can't be used:

```{code-cell} ipython3
from pylimid.solver import is_solvable

is_solvable(carts.snapshot())
```

The scan has no trouble here since each decision has two actions and there is a single situation, so there are only $2 \times 2 = 4$ policies to try:

```{code-cell} ipython3
print(solve(carts).render(carts))
```

Mia should take the beach and Leo the park. Note that the reason backward induction fails is that each decision's best option depends on a choice the other can't see, **which leaves no order in which to resolve them**.

## Fitting a policy with continuous information

A decision with a continuous parent still chooses among its finite actions, but a table cannot list every possible observation. We choose the form of the rule when defining the diagram, and the solver fits its parameters. This choice is part of modeling the decision; the solver does not select an architecture for us.

The first supported family is {py:class}`LinearSoftmax`: linear scores followed by softmax probabilities over the actions. Numerical parents supply raw values (arrays are flattened); categorical parents supply one-hot features in their declared state order. There is no automatic scaling. This family is useful when a linear boundary can describe the action choice; neural networks and custom trainable families are not implemented.

For a small example, suppose we observe a signal uniformly distributed between −1 and 1. Taking action `on` earns the signal's value; `off` earns zero. We declare the family on the decision and select the optimizer explicitly:

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
    method="policy_gradient",
    num_steps=300,
    num_samples=2000,
    learning_rate=0.05,
    rng_key=jax.random.PRNGKey(0),
)
print(fit.render(signal_diagram))
rule = fit.policy["action"]
print("Action probabilities at signal=0.8:", rule.probabilities(signal=jnp.array(0.8)))
```

The returned {py:class}`SoftmaxPolicy` keeps the family, weights, biases, parent bindings and action labels. It remains stochastic: `probabilities` returns probabilities in `states` order, and `sample` takes an explicit random key. Replacing it with the largest-probability action would define a different policy.

The optimizer samples complete scenarios, estimates how changing the action probabilities changes expected utility, and updates the parameters with Adam. NumPyro supplies distributions; PyLIMID constructs the expected-utility gradient with JAX. Utilities and chance distributions must be JAX-traceable, but need not be differentiable with respect to a sampled integer action.

`num_samples` controls the batch size per update and the final evaluation size; `num_steps` controls the training budget. The reported utility uses fresh simulations independent of training. We can also evaluate the fitted policy with a new key:

```{code-cell} ipython3
from pylimid import infer

payoff = infer(
    signal_diagram,
    ["payoff"],
    policy=fit.policy,
    num_samples=10000,
    rng_key=jax.random.PRNGKey(22),
)["payoff"]
print("Fresh expected utility:", payoff.mean())
```

This is an approximate optimization result, with no global or local optimality guarantee, even within the selected family. Compare different training keys and budgets as well as independent utility estimates. More evaluation samples reduce estimation noise, but cannot fix an unsuitable family or incomplete optimization.

### Training several decisions together

For several decisions, we declare a supported family on each one. The solver fits a joint policy: a mapping from decision names to local fitted rules. Each rule receives only its declared information parents. Earlier observations and actions are remembered only when included explicitly as parents.

Suppose the first action changes a later measurement. We add a second decision that sees only that measurement, and a further payoff:

```{code-cell} ipython3
signal_diagram.add_node(
    ChanceNode(
        "reading",
        parents=("signal", "action"),
        dist=lambda signal, action: dist.Delta(signal + action),
    )
)
signal_diagram.add_node(
    DecisionNode(
        "follow_up",
        parents=("reading",),
        states=("off", "on"),
        policy_family=LinearSoftmax(),
    )
)
signal_diagram.add_node(
    UtilityNode(
        "follow_up_payoff",
        parents=("reading", "follow_up"),
        values=lambda reading, follow_up: reading * follow_up,
    )
)
joint_fit = solve(
    signal_diagram,
    method="policy_gradient",
    num_steps=300,
    num_samples=2000,
    rng_key=jax.random.PRNGKey(0),
)
print(joint_fit.render(signal_diagram))
```

`joint_fit.policy` contains rules for `action` and `follow_up`. The second rule sees `reading`; it cannot inspect `signal` or `action` directly. During each training update, the solver samples the first action, generates the resulting reading, then samples the follow-up action. It updates both rules using the sum of all utility nodes. This also supports diagrams with limited memory that cannot be solved by one pass of backward induction. We can pass the complete mapping to `infer(policy=joint_fit.policy)` for fresh forward evaluation.

The solver returns the final training iterate and evaluates it on independent scenarios. It does not select among training checkpoints or report convergence diagnostics. `method="auto"` continues to choose table solvers and rejects a declared family; use `method="policy_gradient"` explicitly. Every decision must declare `LinearSoftmax` or `TabularSoftmax`. Fitted rules support forward evaluation without observed chance-node evidence; custom trainable families and posterior reuse remain future work.

### Trainable tables for categorical information

With {py:class}`TabularSoftmax`, each combination of categorical parent states has independent scores for the available actions. Softmax converts each row's scores to action probabilities. This can express interactions that adding separate linear effects cannot, such as choosing `on` when exactly one of two observations is `yes`:

```{code-cell} ipython3
from pylimid import TabularSoftmax

categorical_diagram = InfluenceDiagram()
for name in ("left", "right"):
    categorical_diagram.add_node(
        ChanceNode(name, states=("no", "yes"), dist=lambda: dist.Bernoulli(0.5))
    )
categorical_diagram.add_node(
    DecisionNode(
        "action",
        parents=("left", "right"),
        states=("off", "on"),
        policy_family=TabularSoftmax(),
    )
)
categorical_diagram.add_node(
    UtilityNode(
        "payoff",
        parents=("left", "right", "action"),
        values=lambda left, right, action: jnp.where(
            action == (left != right), 1.0, 0.0
        ),
    )
)
table_fit = solve(
    categorical_diagram,
    method="policy_gradient",
    num_steps=200,
    num_samples=2000,
)
print(table_fit.render(categorical_diagram))
```

The fitted {py:class}`TabularPolicy` keeps `logits`, parent labels and action labels. The leading axes of `logits` follow `parents` and their state-label order; the last axis follows the action `states`. `probabilities(left=jnp.array(0), right=jnp.array(1))` inspects a row. A parentless decision has a single action-score vector. Every parent must have named finite states and a scalar state index; continuous values and numerical counts without named states cannot index a trainable table. Table size grows as the product of the parents' state counts, and rare combinations may receive little training.

We can also train a table rule alongside a function rule. Here the categorical action affects a later continuous reading, which a {py:class}`LinearSoftmax` follow-up decision observes together with the remembered action:

```{code-cell} ipython3
categorical_diagram.add_node(
    ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0))
)
categorical_diagram.add_node(
    ChanceNode(
        "reading",
        parents=("signal", "action"),
        dist=lambda signal, action: dist.Delta(signal + action),
    )
)
categorical_diagram.add_node(
    DecisionNode(
        "follow_up",
        parents=("reading", "action"),
        states=("off", "on"),
        policy_family=LinearSoftmax(),
    )
)
categorical_diagram.add_node(
    UtilityNode(
        "later_payoff",
        parents=("signal", "follow_up"),
        values=lambda signal, follow_up: signal * follow_up,
    )
)
mixed_fit = solve(
    categorical_diagram,
    method="policy_gradient",
    num_steps=400,
    num_samples=2000,
)
print(mixed_fit.render(categorical_diagram))
```

Both rules train jointly using total utility, and the fitted mapping can be passed to {py:func}`infer` for forward evaluation. The table remains stochastic, including in a fully categorical diagram; choosing each row's largest score creates a different policy. Trainable tables provide an approximate alternative to the scan and backward induction. Backward induction still directly compares finite actions on supported soluble diagrams. Training tables with policy gradients does not combine those solvers or inherit their optimization guarantees.

## When does solve() fail?

{py:func}`solve` reports these unsupported cases:

- **The diagram has no decisions.** In that case there is nothing to optimise, so there is no policy to return.
- **A table solver receives numerical information.** Tables require named finite parent states, so they cannot directly use continuous measurements or numerical counts. We can bin the reading into categories, or use the policy-gradient method above. Numerical variables can still feed other chance nodes and utilities in any solver.
- **The selected solver cannot fit the declared family.** Table solvers reject declared families. Policy-gradient training requires `LinearSoftmax` or `TabularSoftmax` on every decision; it rejects missing families and trainable tables with non-scalar or non-categorical information.

Diagrams without decisions and incompatible table-solver inputs raise {py:class}`~pylimid.InferenceError`. Policy-gradient training raises `ValueError` for a missing family or invalid optimizer settings.
