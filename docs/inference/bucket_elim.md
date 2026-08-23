# Bucket elimination — exact policies for discrete influence diagrams

The graph-native influence-diagram solver in `decisionpy.inference.id`, the counterpart to variable elimination: same factor machinery, but for the full decision objective — maximize expected total utility by choosing, per decision, the best action given its observed information set. Numpy-only, exact, and the primary engine behind `solve()`.

> **Where this fits.** [`decision_node.md`](../graph/decision_node.md) settles the decision-node representation and the solving strategies (intervention-scan vs. policy-as-parameters); this note is about the *implemented* exact solver. It consumes a validated `Snapshot` (see [`diagram.md`](../graph/diagram.md)) and the shared factor primitives in `decisionpy.inference.utils` (same as [`variable elimination`](./backend_numpyro.md)'s sibling in `decisionpy.inference.ve`).

---

## What it computes

`solve(snapshot) -> Solution` returns a `Solution` with:

- **`policy`** — decision name → `{information_set_assignment: action}`: the locally-optimal action for every situation the decision can observe. Information-set variables that don't affect the payoff get the same action for every value.
- **`expected_utility`** — the maximum expected total utility under that policy.

```python
from decisionpy.inference.id import solve

result = solve(diagram.snapshot())
result.policy["treat"]  # {(0,): 0, (1,): 1}  — action per info-set assignment
result.expected_utility  # 78.0
```

The policy is a solve-time object; it is not stored on the `DecisionNode`. Binding it back onto a diagram (clamping every decision to its chosen action) is the `infer(..., policy=...)` step that collapses the ID to a Bayesian network — the unified `solve()` / `infer()` wiring in `decisionpy.inference.engine` dispatches to this solver for all-categorical diagrams.

## Algorithm

1. **Factor build** — one CPT factor per chance node (`P(node | parents)`), one utility factor per utility node (`U(parents)`). The utility factors combine **additively** into a single factor, because the objective is `E[sum_k U_k]` — utilities add where probabilities multiply.
2. **Reverse-topological elimination** (sinks → roots, the Snapshot's order reversed). For each variable:
   - **chance node** — multiply the probability factors mentioning it; if the variable appears in the utility factor, fold the product into it (`sum_X P * U`), otherwise sum it out among the probability factors;
   - **decision node** — **max it out**: for every instantiation of the information set, keep the best action as the policy and the best value as the utility factor.
3. **Combine the remaining scalars** — the maximum expected utility.

## Supported scope

- **All-categorical diagrams only**: every chance and decision node must declare `states` (a `TypeError` otherwise). Continuous decisions and continuous chance nodes need the NumPyro path.
- **Regular / decoupled structures**: reverse-topological elimination is exact when a decision's best action never depends on a variable outside its information set. If it does (a non-regular LIMID — e.g. a utility depending on a hidden chance variable that is not in the decision's information set), `solve` raises `NotImplementedError`; that case needs arc reversal, deferred.
- Complexity is exponential in treewidth, like variable elimination.

## Related

- [`decision_node.md`](../graph/decision_node.md) — decision representation and the two solving strategies (Strategy B: intervention-scan; Strategy A: policy-as-parameters).
- [`utility_node.md`](../graph/utility_node.md) — the utility representation the solver consumes.
- [`inference_strategy.md`](./inference_strategy.md) — why VE stays the exact discrete engine and NumPyro grows as the primary mixed-type engine.
- `decisionpy/inference/id/bucket_elim.py` — module docstring with the algorithm steps.
