# Multiple utility nodes — payoff aggregation

An influence diagram may declare more than one utility node. pylimid aggregates them **additively**: the total utility is the sum of the leaves, and `solve()` maximizes its expectation. This is the standard influence-diagram convention, and it matches pyAgrum. True multi-objective optimization — a Pareto front over vector-valued payoffs — is a different problem and is not implemented.

> **Where this fits.** The single-node representation (`values`, sinks, consistency) is in [`utility_node.md`](./utility_node.md). This note covers what happens when a diagram has several of them.

## The convention

```
maximize  E[ Σ_k U_k(parents(U_k)) ]
```

Because expectation is linear, `E[Σ U_k] = Σ E[U_k]`, so the aggregated objective is exact — no approximation is introduced by summing. Multiple utility nodes therefore mean: *the model asserts additively separable preferences*. A weight for one objective is folded into its `values` callable (multiply the values), not passed to the solver.

When the trade-off is not additive — say `U = min(profit, safety)`, or a product of attributes — the model is a **single** utility node with all attributes as parents; the callable is an arbitrary scalar function of them. The graph layer places no limit on the number of utility nodes, and pylimid utility nodes carry no state label at all (pyAgrum requires exactly one).

## How pylimid computes it

Every solver accumulates the utility leaves along one trajectory and then averages. The trajectory walk is literally `total = total + node.values(...)` — `backward_induction.py`, `batched_scan.py`, and `intervention_scan.py` all do this. `Solution.expected_utility` is one float: the Monte-Carlo estimate of the summed objective. There are no per-utility expected values, and no variance is reported.

## Same as pyAgrum

pyAgrum's `ShaferShenoyLIMIDInference` follows the same convention: with several utility nodes, `MEU()` maximizes the sum. Verified on the toy model below — same optimal action, same value:

| | pylimid (`solve`) | pyAgrum (`MEU`) |
| --- | --- | --- |
| optimal policy | `{D: b}` | `b` |
| expected utility | `2.0` | `2.0` |

The only differences are representational: pyAgrum requires each utility variable to carry exactly one state (pylimid utility nodes have none), and pyAgrum's `MEU()` returns `{mean, variance}` while pylimid reports just the mean. A non-additive trade-off goes in a single utility node's callable in pylimid and in a single node's table in pyAgrum — same modeling move.

## Short example

```python
diag = InfluenceDiagram()
diag.add_node(DecisionNode(name="D", states=("a", "b")))
diag.add_node(UtilityNode(name="U1", parents=("D",), values=lambda D: jnp.array([1.0, 0.0])[D]))
diag.add_node(UtilityNode(name="U2", parents=("D",), values=lambda D: jnp.array([0.0, 2.0])[D]))

solution = solve(diag)   # {D: b}, expected utility 2.0
```

| action | U1 | U2 | total |
| --- | --- | --- | --- |
| a | 1 | 0 | 1 |
| b | 0 | 2 | 2 |

Both libraries pick `b` with expected utility `2.0`. The trade-off between the two objectives is not "visible" to the solver — the additive model has already reduced it to one number, weighting each leaf at 1. Re-weighting is just rescaling the callables: `U1 = [2, 0]`, `U2 = [0, 1]` flips the answer to `a` with utility `2`.

## Real multi-objective optimization (Pareto)

What additive aggregation is *not* is multi-objective decision making. There the payoff is a vector `(E[U₁], …, E[U_m])`, there is no total order over vectors, and the answer is generally a **set** of Pareto-optimal policies — none dominated by another — possibly including stochastic mixtures. That is a different optimization problem, and neither pylimid nor pyAgrum represents it: `Solution` holds one policy and one scalar expected utility.

If Pareto analysis is ever needed, the standard reductions are:

- **Weighted sum.** Scale the utility callables by weights `w_k` and re-solve per weight vector. The existing solvers work unchanged, but only *supported* Pareto points are reachable — a non-convex front contains optimal policies that no weighting selects.
- **ε-constraint.** Maximize one objective subject to the others `≥ ε`. Requires constraints, which the library has no native notion of; a soft penalty inside a callable approximates it.
- **Multi-objective MDP / influence-diagram algorithms.** Value iteration over convex sets of value vectors (convex-hull, Pareto Q-learning). Research-grade; not on the roadmap.

Until then, the additive convention is the whole story: several utility nodes are one objective, summed. If you wanted a Pareto analysis, the sum will hide the trade-off.
