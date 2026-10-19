# Decision nodes — representation strategy

Design note covering how a `DecisionNode` should be represented under the hood in the NumPyro backend, and how discrete, continuous, and mixed decisions compose.

> **Related.** The chance-node distribution representation is settled in [`chance_node.md`](./chance_node.md) (callable-primary, CPT as sugar). The mutability and validation model that governs *all* nodes — including the `UNCONFIGURED`/`STALE`/`CONSISTENT` consistency states and the editing-vs-inference gate — is settled in [`diagram.md`](./diagram.md). A `DecisionNode` inherits that model: it may be added before its information set is fully wired, and its policy is configured after structure is in place.

---

## Representation in the graph layer

`DecisionNode` lives in [`graph/decision_node.py`](../../pylimid/graph/decision_node.py), subclasses the shared [`Node`](../../pylimid/graph/node.py), and participates in the diagram like any other node. What it carries:

- **`parents`** — the **information set**: the variables observed when the decision is made. *Not* a causal dependency. This is the standard influence-diagram convention; the field name is reused from chance/utility nodes so the container's shared bookkeeping (validation, topological ordering, cycle detection) applies unchanged.
- **`states`** — the available actions (labels). `None` marks the decision as not-yet-configured (or, forward-looking, continuous — to be owned by the solver).
- **No `dist`.** The decision's value is chosen, not sampled.

A decision is `CONSISTENT` once its action `states` are declared, and `UNCONFIGURED` before. There is **no STALE state** for a decision: the action space does not depend on the information set's size, so adding or removing an information parent never invalidates it.

The implemented solvers live in `pylimid.solver`. `solve(method="auto")` selects backward induction when the diagram has a valid solution ordering and the batched policy scan otherwise. Both estimate expected utility by forward simulation and require finite decisions with discrete information sets; chance variables may be discrete, continuous, or both in the same diagram. The [solver algorithms](../solver/algorithms.md) describe the costs, solvability criterion, and validation references.

[Discrete decisions with continuous information](../continuous_information/README.md) are proposed separately from continuous actions. The action `states` and information-arc semantics stay the same; the policy representation and optimization method change. This extension can be implemented first.

---

## What "solving a LIMID" means here

A LIMID is solved when each decision node has a *local policy* δᵢ (a function from its observed information set → an action) such that the expected total utility is maximized. There is no forgetting barrier, so each policy is independent — a later decision cannot depend on the full history.

The objective is:

```
maximize   E[ Σ_k  U_k(parents(U_k)) ]
 over δ₁..δₙ
```

where the expectation is over chance variables, whose distributions depend on the chosen decisions through the graph.

---

## Two strategies

The *graph* and *node definitions* are identical in both strategies — only the solver differs.

### Strategy A — policy as parameters (deferred)

In this proposed approach, the solver owns a parameterized policy: for example, logits over finite actions as a function of continuous information. Continuous-action policies would be a separate extension. Solving optimizes an explicit expected-utility objective with an appropriate estimator: score-function gradients for sampled discrete actions, probability-weighted sums for enumerated actions, or suitable pathwise gradients for continuous actions. NumPyro supplies probabilistic machinery; ordinary SVI or a utility `numpyro.factor()` alone does not establish an equivalent maximum-expected-utility objective. The [continuous-information proposal](../continuous_information/README.md#a-general-approximate-solver) develops the finite-action case.

### Strategy B — intervention scan

Decisions are discrete action choices. The solver enumerates the policy space — one action per information-set assignment per decision — and estimates each policy's expected utility by forward-sampling the diagram with every decision resolved deterministically from its observed information set (a Monte-Carlo estimate per policy). The policy with the highest estimated EU wins. Implemented as `pylimid.solver.scan.intervention.solve`; `solve(method="scan")` uses the batched implementation in `pylimid.solver.scan.batched`. The full solver family — the scan, the batched scan, and the backward-induction / SPU path — and the papers behind them are in [solver algorithms](../solver/algorithms.md).

### Trade-offs

|                     | A. Policy as params            | B. Intervention scan             |
| ------------------- | ------------------------------ | -------------------------------- |
| Best for            | discrete decisions with continuous information; future continuous actions | finite actions and categorical information sets |
| Optimizes via       | gradients (fast, scales)       | enumeration / search (exact-ish) |
| Main difficulty     | gradient of `E[U]` w.r.t. policy — SVI's ELBO framing does not directly fit EU maximization, so a custom loss around `jax.grad` + trace is required | exponential blow-up in (#decisions × action-space) |
| Verifiability       | hard to tell convergence from correctness | trivially correct — EU measured per action |

Bucket elimination was the original exact solver and was retired with the other exact engines; the [NumPyro-only ADR](../ADR/25_08_2026_numpyro_only_engine.md) records that decision. Solver choice does not change the graph abstraction: a `DecisionNode` declares its actions and the parents it can observe.

---

## Mixed discrete + continuous decisions

Mixed discrete and continuous decisions remain a future proposal. The intended approach nests discrete policy search around continuous policy optimization; it is not supported by the current solvers.

Standard decomposition (same structure as mixed-integer nonlinear programming):

> Enumerate over the discrete decisions; for each discrete setting, solve the continuous subproblem by gradient. Pick the discrete setting whose best continuous response scores highest.

Formally, for a discrete decision $D_d \in \{a, b, c\}$ and a continuous decision $D_c \in \mathbb{R}$:

$$\max_{D_d, D_c} \mathbb{E}[U] \;=\; \max_{D_d \in \{a,b,c\}} \; \max_{D_c} \mathbb{E}[U \mid D_d]$$

The outer loop is Strategy B (implemented); the inner loop is Strategy A (deferred). The mixed case is just composition.

### Concrete shape

```
for action in discrete_actions:                      # outer: B
    intervened = numpyro.handlers.do(model, {d: action})
    cont_opt = maximize_expected_utility(intervened)  # inner: A (jax.grad on a param)
    best_EU[action] = expected_utility(intervened, cont_opt)
winner = argmax(best_EU)
```

### Where it stops being clean

- **Non-convex continuous subproblems.** Gradient ascent finds *a* solution per branch, not guaranteed the global one. Mitigations: random restarts, or exploit convex substructure (e.g. LQG) when present.
- **Multiple discrete decisions with interaction.** Two interdependent discrete decisions enumerate a product space — exponential blow-up. LIMID structure (local information sets, no forgetting barrier) helps but does not eliminate it. No free lunch here; classical ID solvers hit the same wall.

---

## Design implication

When Strategy A lands, the mixed case forces the "pluggable per-node" structure:

- A `DecisionNode` knows its own type (discrete / continuous) and exposes the right solving primitive — enumeration for discrete, gradient for continuous.
- A `Solver` walks the graph and dispatches per decision: discrete decisions become outer enumeration branches, continuous decisions become inner optimizations.
- The mixed case is not special-cased — it falls out of composing the two dispatch types.

The asset is the **per-node dispatch abstraction**; the strategies (enum-scan vs. policy-gradient) are interchangeable parts behind it.
