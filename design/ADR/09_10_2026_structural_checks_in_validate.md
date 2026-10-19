# Report structural problems in `validate()` only

**Date:** 2026-10-09
**Status:** Settled
**Supersedes:** Principle 4 of [`diagram.md`](../graph/diagram.md) as it stood before this date ("structural invariants are enforced eagerly on the happy path, defensively elsewhere").

> **Where this fits.** This is a *decision record*. The checking model itself is described in [`diagram.md`](../graph/diagram.md), Principles 2 and 4.

---

## The decision

Cycles and utility nodes with children are reported by `InfluenceDiagram.validate()` and are no longer rejected by `InfluenceDiagram.add_edge`. `add_edge` also accepts a parent that is not in the diagram yet, the same dangling reference that `add_node` already allowed. Editing methods only refuse requests they cannot carry out: a duplicate name, an edge into a child that is not in the diagram, or a malformed field (empty name, a node listing itself as a parent).

The `CYCLE` problem now names the nodes on the cycle, in edge order (for example `'b' -> 'c' -> 'a' -> 'b'`), and its `node` field is a node on the cycle instead of `"*"`.

---

## Why

The eager checks only covered one of three ways to create an edge. `add_node` with `parents=` and a node's own `add_parent` both skipped them, so whether a cycle or a utility-with-child was rejected depended on how and in which order the diagram was built. `validate()` already had to check both invariants for the other two paths.

Checking in one place gives one rule that does not depend on construction order, and it keeps incremental edits possible that the eager check blocked, such as reversing an arc by adding the new one before removing the old one. It also matches the mutable-workspace model ([`17_07_2026_mutability_design_decision.md`](./17_07_2026_mutability_design_decision.md)): the diagram may be unsound while it is being edited, and `validate()` / `snapshot()` are the gate.

The cost is that a cycle is reported at `validate()`, `solve()` or `infer()` rather than at the line that created it. Naming the cycle's nodes in the message is meant to make that cheap to trace back. pyAgrum and pgmpy reject cycles when the arc is added, so some users will expect the eager behaviour.

---

## What changed in the tree

- `pylimid/graph/diagram.py`: `add_edge` keeps only the child-existence check (the self-loop check is the node's own field validation); the private `_reaches` and `_children_of` helpers are removed; `validate()` finds and names a cycle through `_kahn_order` and `_find_cycle`.
- `tests/graph/test_diagram.py`: tests that expected `add_edge` to raise now expect `validate()` to report the problem; new tests cover a dangling parent through `add_edge`, the cycle message, and reversing an arc.
- `design/graph/diagram.md`, `design/graph/utility_node.md`, `design/README.md`, the module docstrings and `docs/user_guide/utilities.md` describe the new rule.
