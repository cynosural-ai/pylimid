---
file_format: mystnb
kernelspec:
  name: python3
---

# Solving

`solve(diagram, method=..., num_samples=..., rng_key=...)` returns the optimal policy and the expected utility it achieves. It is the front door to the solver family: two Monte-Carlo solvers with different guarantees, chosen automatically by default.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve
from pylimid.inference.numpyro.solvers import is_solvable, solvability_order
```

## The Solution

A `Solution` has three fields:

- `policy` — one action per information-set assignment, per decision. Keys are tuples of parent-state indices, in parent order (see `Decisions and information sets`).
- `expected_utility` — a Monte-Carlo estimate of the expected total utility under that policy.
- `method` — the solver that produced it: `"backward_induction"` or `"scan"`, never `"auto"`. The two solvers carry different guarantees, so the result records which one ran.

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
```

## The solvability gate

A LIMID need not admit a clean solving order. The diagram is **solvable** when there is an ordering in which each decision can be finalized from its information set once the later decisions are resolved — the exact one-pass condition for backward induction. `is_solvable` answers that, and `solvability_order` returns an order when one exists:

```{code-cell} ipython3
is_solvable(medical.snapshot()), solvability_order(medical.snapshot())
```

This model is solvable, so the default `method="auto"` picks backward induction:

```{code-cell} ipython3
solution = solve(medical, num_samples=20000)
solution.method, solution.policy, round(solution.expected_utility, 1)
```

Backward induction resolves one decision at a time, from the last to the first, estimating each action's continuation value per information-set cell by stratified forward sampling. Its cost is **additive in the decisions**, and on a solvable diagram it is optimal up to sampling noise.

The alternative is the scan, which enumerates every policy and estimates each one's expected utility by forward sampling, keeping the best:

```{code-cell} ipython3
scan = solve(medical, method="scan", num_samples=20000)
scan.method, scan.policy, round(scan.expected_utility, 1)
```

The scan handles any diagram — soluble or not — but its cost grows exponentially with the policy space, which itself is exponential in the number of information-set assignments. `"auto"` uses it exactly when backward induction cannot run, and `method="backward_induction"` on a non-solvable diagram raises `InferenceError` rather than guessing.

## Monte-Carlo error

Both solvers produce estimates, not exact answers. `num_samples` is the number of forward trajectories per evaluation — per candidate policy for the scan, per action per information set for backward induction — and the run is deterministic for a fixed `rng_key` (the default is seed 0). The same model at 2000 samples:

```{code-cell} ipython3
for seed in (0, 1, 2):
    estimate = solve(medical, rng_key=jax.random.PRNGKey(seed))
    print(seed, round(estimate.expected_utility, 2))
```

The spread is the honest cost of the estimator. Raise `num_samples` to shrink it; pass `rng_key` to reproduce or vary a run.

## Failure modes

- **No decisions.** `solve` requires at least one decision node; a Bayesian network is queried with `infer`.
- **Continuous information sets.** Decisions and their information sets must be discrete — a continuous parent cannot be tabulated, and the solvers reject it.
- **Unreachable information-set cells.** Backward induction needs a sampled continuation value for every cell of every decision's information set. A structurally impossible cell — often a deterministic "not applicable" state — leaves one unsampled and raises; drop the state or solve with `method="scan"` (the oil-field tutorial walks through this).
- **Non-traceable utilities.** `values` callables run inside vectorized JAX passes, so pure `jnp` expressions only (see `Utilities`).
