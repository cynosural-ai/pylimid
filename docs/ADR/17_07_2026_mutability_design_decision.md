# Commit to the mutable workspace; retire build-once

**Date:** 2026-07-17
**Status:** Settled
**Supersedes:** the parallel build-once prototype that lived alongside the mutable one.

> **Where this fits.** This is a *decision record* — it records the call and what changed in the tree. The *reasoning* behind the mutable model itself is in [`diagram.md`](../diagram.md); this note does not re-argue it.

---

## The decision

The influence diagram is a **mutable workspace**: nodes may be added before their parents, edges rewired, and distributions filled in at any time, with inference gated behind an explicit `validate()` / `snapshot()` checkpoint. This is now the library's sole container design.

The alternative — a **build-once** container that required every parent to be present at registration and demanded a complete, valid object in a single atomic step — is rejected and removed from the tree.

The rationale (incremental construction by external authors — scripts, UIs, LLMs — that can never produce a complete diagram atomically) is settled in [`diagram.md`](../diagram.md); the seven principles there govern every node and the container.

---

## What changed in the tree

The two designs previously coexisted as parallel prototypes under `src/decisionpy/graph/`:

| Before                                   | After                              |
| ---------------------------------------- | ---------------------------------- |
| `chance_node.py` (frozen / build-once)   | removed                            |
| `diagram.py` (frozen / build-once)       | removed                            |
| `chance_node_mutable.py`                 | renamed → `chance_node.py`         |
| `diagram_mutable.py`                     | renamed → `diagram.py`             |
| `tests/test_chance_node.py` (frozen)     | removed                            |
| `tests/test_diagram.py` (frozen)         | removed                            |
| `tests/test_chance_node_mutable.py`      | renamed → `test_chance_node.py`    |
| `tests/test_diagram_mutable.py`          | renamed → `test_diagram.py`        |

### Why rename rather than keep the `_mutable` suffix

Once the build-once alternative is gone, the suffix has nothing to contrast against — it would enshrine a distinction that no longer exists in the code. The mutable container *is* the container, so it takes the canonical names `chance_node.py` and `diagram.py`. Callers import from `decisionpy.graph` exactly as they would for any single-implementation library.

### Why remove the build-once files rather than keep them as reference

A reference implementation that nothing imports, nothing tests, and that encodes a design we have explicitly rejected is a liability: it drifts, it misleads readers about the intended direction, and it tempts confusion about which file is canonical. The rejected design is fully documented in [`diagram.md`](../diagram.md) ("What this deliberately is not") and can be recovered from version control if ever needed. Keeping it in the tree served no active purpose.

---

## Verification

`graph/__init__.py` exports `ChanceNode` and `InfluenceDiagram` from the canonical modules; the renamed test suites pass against them.

---

## Open after this decision

This settles the *container*. It does not advance:

- the NumPyro translator (not started),
- `from_cpt` sugar (deferred — see [`chance_node.md`](../chance_node.md)),
- parametric learning / `fit` (deferred — see [`diagram.md`](../diagram.md)),
- decision and utility nodes (later milestones).
