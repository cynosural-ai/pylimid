---
file_format: mystnb
kernelspec:
  name: python3
---

# Decisions and information sets

A `DecisionNode` is a variable the agent controls. It has no distribution. It carries two things: an **information set** (its `parents` — the variables observed when the choice is made) and an **action space** (`states` — the available choices).

The information set is structural, not causal. The same `parents` tuple that expresses "who influences whom" for a chance node means "what is known at decision time" here.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve
```

## LIMID semantics: memory is explicit

A `pylimid` diagram makes no no-forgetting assumption. A decision observes *exactly* its parents — nothing implicit. If a later decision needs to remember an earlier decision or observation, you draw that arc yourself:

```{code-cell} ipython3
test_treat = InfluenceDiagram()
test_treat.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
test_treat.add_node(DecisionNode(name="test", states=("do", "skip")))
test_treat.add_node(
    ChanceNode(
        name="result",
        parents=("disease", "test"),
        states=("negative", "positive"),
        dist=lambda disease, test: dist.Categorical(
            probs=jnp.array(
                [
                    [[0.9, 0.1], [0.5, 0.5]],  # healthy
                    [[0.1, 0.9], [0.5, 0.5]],  # sick
                ]
            )[disease, test]
        ),
    )
)
test_treat.add_node(
    DecisionNode(name="treat", parents=("test", "result"), states=("yes", "no"))
)
test_treat.add_node(
    UtilityNode(
        name="utility",
        parents=("disease", "treat"),
        values=lambda disease, treat: jnp.where(
            disease == 1,
            jnp.where(treat == 0, 80.0, -50.0),
            jnp.where(treat == 0, -20.0, 0.0),
        ),
    )
)
test_treat.validate()
```

Here the `test → treat` arc is a **memory arc**: the treatment decision remembers whether the test was run, on top of observing its result. Drop that arc and `treat` observes only `result` — a different decision problem, not a solver trick. More memory can only weakly improve the achievable expected utility, at the cost of larger information sets.

```{code-cell} ipython3
test_treat["treat"].parents
```

## Actions

`states` is the action space, and like a chance node it is a tuple of labels whose indices are the values that flow through the graph. A decision is `UNCONFIGURED` until `states` are set; unlike a chance node there is no `STALE` state, because the action space does not depend on the information set — adding a parent never invalidates it.

Continuous decisions are not supported yet: `states=None` leaves the node unconfigured, and the solvers reject decisions whose information sets contain continuous variables.

## What the policy looks like

The policy is where the information set becomes concrete. `solve` returns one action per **information-set assignment**, encoded as a tuple of parent-state indices in parent order:

```{code-cell} ipython3
solution = solve(test_treat, num_samples=20000)
solution.policy
```

The `test` key `()` is the empty information set. The `treat` keys are `(test, result)` pairs: `(0, 0)` is "the test ran and came back negative", `(0, 1)` is positive, and `(1, 0)`/`(1, 1)` are the two filler rows when the test was skipped. The optimizer covers every cell it can reach, even if the policy would never take that branch. (We cover the solvers themselves on the Solving page.)

Two consequences worth remembering:

- **Parent order matters for reading the policy.** The tuple follows the order the parents are declared in.
- **Unreachable cells can bite.** Backward induction needs a sampled continuation value for every cell; a structurally impossible cell (a deterministic "not applicable" result, say) forces the scan instead. The oil-field tutorial shows the modeling workaround.
