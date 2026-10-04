---
file_format: mystnb
kernelspec:
  name: python3
---

# Utilities

A `UtilityNode` is a deterministic payoff, `U(parents)`. Its `values` is an ordinary Python callable that receives the resolved parent values as keyword arguments and returns a scalar. There is no table requirement and no distribution: any JAX-traceable expression of the parents is a valid utility.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve
```

## The payoff is a function

The medical model is a compact example: recovering is worth 100, treating costs 20. Expressing those as two separate utility nodes is idiomatic — the total utility is the sum of every utility node in the diagram.

```{code-cell} ipython3
medical = InfluenceDiagram()
medical.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
medical.add_node(
    DecisionNode(name="treat", parents=("disease",), states=("no", "yes"))
)
medical.add_node(
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
medical.add_node(
    UtilityNode(
        name="benefit", parents=("recovery",), values=lambda recovery: 100.0 * recovery
    )
)
medical.add_node(
    UtilityNode(name="cost", parents=("treat",), values=lambda treat: -20.0 * treat)
)
medical.validate()
```

Splitting the payoff like this keeps each node simple and reusable — the cost node knows nothing about recovery, and the benefit node nothing about treatment. The solver only ever sees their sum.

```{code-cell} ipython3
solution = solve(medical, num_samples=20000)
solution.policy, round(solution.expected_utility, 1)
```

That is the same optimal policy as the quickstart's single-utility version, with an expected utility that agrees up to Monte-Carlo error. Multiple utility nodes are a modelling convenience, not a different objective.

## Beyond money

`values` is any function, so utilities are where risk preferences live. The tutorials assume risk neutrality (utility proportional to money), but nothing in the library does: a concave transform of a monetary outcome, or a utility that saturates, is just another expression. What the library *does* require is that the expression is traceable.

## Utilities are sinks

A payoff is terminal: a utility node may not have children. `add_edge` rejects an edge whose parent is a utility node, and `validate()` backstops a node mutated directly. Like every node, a utility is `UNCONFIGURED` until `values` is set, and `STALE` if its signature no longer matches its parents after an edge change.

## The callable runs inside JAX

The same rule as chance-node distributions applies: `values` is traced and evaluated inside vectorized forward passes, so it must be a pure `jnp` expression — no `float()` / `int()` coercion, no Python `if` on a traced value. Branch on values with `jnp.where`:

```{code-cell} ipython3
def utility(disease, treat):
    """A saturating payoff: 100 when sick, independent of treatment."""
    return jnp.where(disease == 1, 100.0, 0.0) * (1 - treat)
```

One consequence worth noting: `infer` ignores utility nodes entirely. When every decision is bound by a policy, the influence diagram collapses to a Bayesian network and utilities play no part in the posterior.
