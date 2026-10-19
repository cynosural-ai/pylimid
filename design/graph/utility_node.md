# Utility nodes — the payoff representation

Design note covering how a `UtilityNode` is represented in the graph layer, how its consistency model works, and why utility nodes are *sinks*.

> **Where this fits.** The [diagram principles](diagram.md) govern mutation, consistency, and validation for all nodes. This note covers payoff representation and aggregation; [decision nodes](decision_node.md) and the [solver algorithms](../solver/algorithms.md) cover optimization. The shared consistency model lives on [`Node`](../../pylimid/graph/node.py).

---

## The decision

**Canonical form: a callable `values(parent_assignments) -> float`.** A utility node has no distribution and no states — it is never sampled and carries no outcomes. It is a deterministic scalar function of its parents:

```python
UtilityNode(
    name="payoff",
    parents=("rain", "invest"),
    values=lambda rain, invest: revenue[rain, invest] - cost[invest],
)
```

The `values` callable receives resolved parent values as **keyword arguments** keyed by parent name — the same convention as a chance node's `dist`. This makes the node robust to parent reordering and keeps the graph layer's calling convention uniform across node types.

`values=None` (the default) marks the node as not-yet-configured — a legitimate intermediate state while the diagram is being built.

---

## Why a callable (not a table)

The same argument as for chance nodes (see [`chance_node.md`](./chance_node.md)): a table over discrete parents is always recoverable from a callable, but the reverse is not true, and a table cannot represent a utility that depends on a *continuous* parent. The callable-primary form keeps the "mixed-type" premise of the library intact — a utility can depend on continuous rainfall just as easily as on a binary decision.

For a purely discrete utility, the table is simply an array indexed by the parents' integer values, wrapped in a callable — no separate node type.

---

## Utility nodes are terminal (sinks)

In an influence diagram a payoff is always a **sink**: it has parents but no children. Like acyclicity ([`diagram.md`](./diagram.md), Principle 4), this is checked in `InfluenceDiagram.validate`, which reports a utility node with a child as a `DiagramProblemKind.UTILITY_NOT_SINK` problem, whether the arc came from `add_arc`, `parents=` in `add_node`, or a node's own `add_parent`.

The check reads `node.is_sink` rather than branching on node type, so the container stays free of per-type logic.

---

## Multiple utilities and weighting

Several utility nodes assert additively separable preferences. Every solver adds their values along each sampled trajectory and maximizes the mean total:

```
maximize E[sum_k U_k(parents(U_k))]
```

Linearity of expectation means summing utilities introduces no additional approximation. A weight belongs inside the relevant `values` callable; the solver takes no separate objective weights. `Solution.expected_utility` reports one estimated total, without per-utility means or a variance. This is also pyAgrum's additive convention.

```python
import jax.numpy as jnp
from pylimid import DecisionNode, InfluenceDiagram, UtilityNode, solve

diagram = InfluenceDiagram()
diagram.add_node(DecisionNode(name="D", states=("a", "b")))
diagram.add_node(
    UtilityNode(name="U1", parents=("D",), values=lambda D: jnp.array([1.0, 0.0])[D])
)
diagram.add_node(
    UtilityNode(name="U2", parents=("D",), values=lambda D: jnp.array([0.0, 2.0])[D])
)
solution = solve(diagram)
```

Action `a` scores `1 + 0 = 1`; action `b` scores `0 + 2 = 2`, so the optimal action is `b`. Rescaling the callables to `U1 = [2, 0]` and `U2 = [0, 1]` makes `a` optimal instead.

For a non-additive trade-off, such as a minimum or product of attributes, put the complete scalar expression in one utility callable with the required parents. The functions must still be JAX-traceable.

A Pareto front over vector-valued payoffs is a separate optimization problem and remains out of scope. Re-solving with different weights is possible today, but weighted sums do not recover every Pareto point of a non-convex front.

---

## Consistency model

`UtilityNode` reuses the shared consistency gate from `Node`, applied to its `values` field (mirroring how `ChanceNode` applies it to `dist`):

| State          | Condition                                             |
| -------------- | ----------------------------------------------------- |
| `UNCONFIGURED` | `values is None`                                      |
| `CONSISTENT`   | `values` names every parent as an explicit parameter  |
| `STALE`        | `values` is set but its signature ≠ `parents`         |

STALE arises, exactly as for chance nodes, when a parent is added or removed after `values` was configured, or when the callable hides a parent behind a variadic `*args` / `**kwargs` (a `**kwargs`-only callable is `CONSISTENT` only for an empty parent set). The wording of the non-CONSISTENT message is owned by the node (`UtilityNode.consistency_message`), so each node type describes its own configurable field.

---

## What is deliberately deferred

- **No policy-as-parameters solving (Strategy A).** The graph layer represents the utility; the solvers evaluate it by forward sampling (see [solver algorithms](../solver/algorithms.md)). Continuous information sets and continuous decisions — the Strategy A path — are deferred.
- **No table sugar** for discrete utilities (analogous to the deferred `from_cpt` for chance nodes). A discrete utility is a callable wrapping an array today.
