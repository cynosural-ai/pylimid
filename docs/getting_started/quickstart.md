---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

This page walks a small influence diagram through its whole lifecycle: build the model, validate it, solve it for the optimal policy, and infer a posterior under that policy.

The model is a medical decision. A patient is healthy or sick; the doctor decides whether to treat; the patient may recover. A recovery is worth 100 and treating costs 20.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    UtilityNode,
    infer,
    solve,
)

diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
diagram.add_node(
    DecisionNode(name="treat", parents=("disease",), states=("no", "yes"))
)
diagram.add_node(
    ChanceNode(
        name="recovery",
        parents=("disease", "treat"),
        states=("no", "yes"),
        dist=lambda disease, treat: dist.Categorical(
            probs=jnp.array(
                [
                    [[0.10, 0.90], [0.08, 0.92]],  # healthy
                    [[0.60, 0.40], [0.20, 0.80]],  # sick
                ]
            )[disease, treat]
        ),
    )
)
diagram.add_node(
    UtilityNode(
        name="utility",
        parents=("recovery", "treat"),
        values=lambda recovery, treat: 100.0 * recovery - 20.0 * treat,
    )
)
```

`disease` and `recovery` are chance nodes, `treat` is a decision, and `utility` is the payoff. A decision's parents are its *information set*: here the doctor decides knowing the diagnosis, so `treat` lists `disease` as a parent.

## Validate

A diagram is a mutable workspace that may be incomplete while you build it. `validate()` collects every problem that would block inference — dangling parents, cycles, unconfigured nodes — and returns them as a list. An empty list means the diagram is sound.

```{code-cell} ipython3
diagram.validate()
```

## Solve

`solve()` returns a `Solution` with the optimal policy and the expected utility it achieves. Both are Monte-Carlo estimates.

```{code-cell} ipython3
solution = solve(diagram)
solution.policy
```

```{code-cell} ipython3
round(solution.expected_utility, 2)
```

The policy maps each decision to an information-set assignment. `treat` has one parent, `disease`, so each key is a one-element tuple of state indices: `(0,)` is `healthy`, `(1,)` is `sick`, and the value is the chosen action index. Reading `solution.policy` above: do not treat a healthy patient, treat a sick one. That matches the exact expected utilities — not treating is worth 90 versus 92 - 20 = 72 when healthy, treating is worth 60 versus 40 when sick — so the expected utility is the prior-weighted average `0.6 * 90 + 0.4 * 60 = 78`.

## Infer

Once every decision is bound by a policy, the influence diagram collapses to a Bayesian network and `infer()` applies. Ask for the posterior over `recovery` under the decision to treat.

```{code-cell} ipython3
result = infer(diagram, query=["recovery"], policy={"treat": 1})
result["recovery"].marginal()
```

The result is `[P(recovery = no), P(recovery = yes)]`. Treating raises the chance of recovery from the baseline because it helps most when the patient is sick.

Binding *every* decision is required — a partially bound diagram is still ambiguous, and `infer()` raises `InferenceError` if a decision is left out. To choose the actions first, call `solve()` and pass its policy to `infer()`.
