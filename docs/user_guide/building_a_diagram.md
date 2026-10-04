---
file_format: mystnb
kernelspec:
  name: python3
---

# Building a diagram

An influence diagram in `pylimid` is a **mutable workspace**, not a build-once container. You add nodes before their parents exist, wire and unwire edges freely, and configure distributions, action spaces, and utilities as you go. Nothing is enforced until you ask for it: inference is gated behind an explicit validation checkpoint.

A Bayesian network is simply a diagram that contains only chance nodes. The moment you add a decision or a utility node, it becomes an influence diagram.

## The three node kinds

Every node carries a `name` and a `parents` tuple, but the meaning of `parents` depends on the kind:

- `ChanceNode` — a random variable `P(name | parents)`; `parents` are a statistical dependency.
- `DecisionNode` — a variable the agent controls; `parents` are its **information set** (what is observed when choosing), and `states` are the available actions.
- `UtilityNode` — a deterministic payoff `U(parents)`; always a sink.

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

diag = InfluenceDiagram()
diag.add_node(
    ChanceNode(
        name="market",
        states=("down", "up"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.4, 0.6])),
    )
)
diag.add_node(DecisionNode(name="invest", parents=("market",), states=("no", "yes")))
diag.add_node(
    UtilityNode(
        name="profit",
        parents=("invest",),
        values=lambda invest: jnp.where(invest == 1, 100.0, 0.0),
    )
)
diag.validate()
```

A node's parents need not exist yet — dangling references are tolerated while you build. `add_edge(parent, child)` is stricter: both endpoints must already be in the diagram, the edge is rejected eagerly if it would create a cycle, and a utility node may never become a parent (a payoff is terminal). Duplicate edges are a no-op, and `remove_node` scrubs the removed name from every survivor's parent set.

The diagram keeps the derived adjacency up to date, so you can introspect it at any point:

```{code-cell} ipython3
diag.names, diag.parents_of("invest"), diag.children_of("market"), diag.topological_sort()
```

## Consistency, not enforcement

Each node tracks its own **consistency** with respect to its parents. Consistency is computed on demand from the live field values and has three states:

- `UNCONFIGURED` — the configurable field is unset (`dist` for chance, `values` for utility, action `states` for decision).
- `STALE` — the field is set, but its signature no longer accepts the current parents. This is what happens after you wire a new parent to an already-configured node.
- `CONSISTENT` — the field accepts the parents; the node is ready.

Field-level mistakes (an empty name, a duplicate parent, a non-callable `dist`) are rejected the instant you set them. Cross-field consistency is *not* enforced on assignment — an inconsistent node is a legitimate intermediate state while a script, a UI, or an LLM is still building the model.

```{code-cell} ipython3
workspace = InfluenceDiagram()
workspace.add_node(
    ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
    )
)
workspace.add_node(
    ChanceNode(name="wet_grass", parents=("rain",), states=("dry", "wet"))
)
workspace.validate()
```

The problem tells you which node and what is wrong. Once we configure the distribution, the workspace is sound:

```{code-cell} ipython3
workspace["wet_grass"].dist = lambda rain: dist.Categorical(
    probs=jnp.array([[0.9, 0.1], [0.2, 0.8]])[rain]
)
workspace.validate()
```

Now wire a new parent to the configured node. Its `dist` only names `rain`, so adding `sprinkler` makes it stale — the diagram does not silently guess what the extra parent means:

```{code-cell} ipython3
workspace.add_node(
    ChanceNode(
        name="sprinkler",
        states=("off", "on"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
workspace.add_edge("sprinkler", "wet_grass")
workspace.validate()
```

Reconfigure the callable to name both parents, and the node becomes consistent again:

```{code-cell} ipython3
workspace["wet_grass"].dist = lambda rain, sprinkler: dist.Categorical(
    probs=jnp.array(
        [
            [[0.95, 0.05], [0.10, 0.90]],  # rain = no
            [[0.80, 0.20], [0.02, 0.98]],  # rain = yes
        ]
    )[rain, sprinkler]
)
workspace.validate()
```

## Validate and snapshot

`validate()` never raises on the first problem: it collects every issue as a `DiagramProblem` — a `kind`, the `node` it belongs to, and a human-readable `message`. An empty list means the diagram is sound. The kinds cover dangling parents, cycles, unconfigured and stale nodes, and utility nodes that acquired a child.

Inference and solving do not read the live workspace directly. They read a `snapshot()`:

```{code-cell} ipython3
snap = workspace.snapshot()
snap.order
```

`snapshot()` validates and freezes a topologically ordered view of the nodes (`parents` first). It is a *logical* snapshot: it holds references to the still-mutable nodes, so editing the diagram afterwards invalidates it. Take a fresh snapshot after editing.

```{admonition} Optional check: does a distribution use its parent?
:class: tip
`validate()` can only inspect the *signature* of a callable, so a table indexed with the wrong parent — the classic mistake where JAX clamps out-of-range indices and silently returns the last row — passes. `probe_discrete_parents()` runs the callables and varies each discrete parent across its states; if the output never changes, you get a `DIST_IGNORES_PARENT` warning. It is opt-in because it executes your code, and it is a warning because a deliberately independent node looks the same. See [](chance_nodes.md) for the details.
```
