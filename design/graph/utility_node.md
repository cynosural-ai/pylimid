# Utility nodes — the payoff representation

Design note covering how a `UtilityNode` is represented in the graph layer, how its consistency model works, and why utility nodes are *sinks*.

> **Where this fits.** This is the utility-side companion to [`decision_node.md`](./decision_node.md): together they complete the three node types of an influence diagram. The *mutability and validation model* — editable fields, the `UNCONFIGURED`/`STALE`/`CONSISTENT` consistency states, the editing-vs-inference gate — is settled in [`diagram.md`](./diagram.md) and lives on the shared `Node` base (see [`node.py`](../pylimid/graph/node.py)). The *solving strategy* that consumes utilities is in [`decision_node.md`](./decision_node.md); this note covers only the node's representation. Several utility nodes aggregate additively — the convention and the contrast with true multi-objective (Pareto) optimization are in [`multiple_utility_nodes.md`](./multiple_utility_nodes.md). Expected-utility solving is implemented by the NumPyro intervention scan (`pylimid.inference.numpyro.solvers`); bucket elimination ([`bucket_elim.md`](../inference/bucket_elim.md)) served as the v0 exact solver and is retired.

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

In an influence diagram a payoff is always a **sink**: it has parents but no children. This is enforced with the same two-layer treatment as acyclicity ([`diagram.md`](./diagram.md), Principle 4):

- **Eagerly** in `InfluenceDiagram.add_edge`: an edge whose *parent* is a utility node is rejected immediately. `add_edge(parent, child)` makes `parent` gain a child, so if `parent.is_sink` is true, the call raises. A utility-with-child is never a useful intermediate state.
- **Defensively** in `InfluenceDiagram.validate`: a node can be mutated directly through its own `add_parent`, bypassing `add_edge`, so a utility node may nonetheless appear as some other node's parent. `validate` reports this as a `DiagramProblemKind.UTILITY_NOT_SINK` problem — the robust backstop.

The check reads `node.is_sink` rather than branching on node type, so the container stays free of per-type logic.

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

- **No policy-as-parameters solving (Strategy A).** The graph layer represents the utility; the NumPyro intervention scan (Strategy B from [`decision_node.md`](./decision_node.md)) evaluates it by forward sampling per candidate policy. Continuous information sets and continuous decisions — the Strategy A path — are deferred.
- **No table sugar** for discrete utilities (analogous to the deferred `from_cpt` for chance nodes). A discrete utility is a callable wrapping an array today.
