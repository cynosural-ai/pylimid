# Decision nodes — representation strategy

Design note covering how a `DecisionNode` should be represented under the hood in the NumPyro backend, and how discrete, continuous, and mixed decisions compose.

> **Related.** The chance-node distribution representation is settled in [`chance_node.md`](./chance_node.md) (callable-primary, CPT as sugar). The mutability and validation model that governs *all* nodes — including the `UNCONFIGURED`/`STALE`/`CONSISTENT` consistency states and the editing-vs-inference gate — is settled in [`diagram.md`](./diagram.md). A `DecisionNode` inherits that model: it may be added before its information set is fully wired, and its policy is configured after structure is in place.

---

## Representation in the graph layer

`DecisionNode` lives in [`graph/decision_node.py`](../decisionpy/graph/decision_node.py), subclasses the shared [`Node`](../decisionpy/graph/node.py), and participates in the diagram like any other node. What it carries:

- **`parents`** — the **information set**: the variables observed when the decision is made. *Not* a causal dependency. This is the standard influence-diagram convention; the field name is reused from chance/utility nodes so the container's shared bookkeeping (validation, topological ordering, cycle prevention) applies unchanged.
- **`states`** — the available actions (labels). `None` marks the decision as not-yet-configured (or, forward-looking, continuous — to be owned by the solver).
- **No `dist`.** The decision's value is chosen, not sampled.

A decision is `CONSISTENT` once its action `states` are declared, and `UNCONFIGURED` before. There is **no STALE state** for a decision: the action space does not depend on the information set's size, so adding or removing an information parent never invalidates it.

The *solving* of decisions is implemented: the **NumPyro intervention scan** (Strategy B, `decisionpy.inference.numpyro.solvers`) — `solve()` enumerates the discrete policy space (one action per information-set assignment per decision) and estimates each policy's expected utility by forward sampling with every decision resolved from its observed information set, keeping the best. It handles mixed and continuous diagrams, and requires discrete information sets. Strategy A (policy-as-parameters) remains the deferred path for continuous decisions and continuous information sets. Bucket elimination ([`bucket_elim.md`](../inference/bucket_elim.md)) served as the v0 exact solver and is retired; its full history is in git. The node's representation is settled here; the solver options are what the sections below are about.

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

### Strategy A — policy as parameters

A decision node is backed by `numpyro.param()` forming a parameterized policy (e.g. logits over actions, or a continuous policy mean/variance). Utility is pushed into the objective as a log-density factor via `numpyro.factor()`. Solving is gradient ascent on `E[U]` (reparameterization or score function).

### Strategy B — intervention scan

Decisions are discrete action choices. The solver enumerates the policy space — one action per information-set assignment per decision — and estimates each policy's expected utility by forward-sampling the diagram with every decision resolved deterministically from its observed information set (a Monte-Carlo estimate per policy). The policy with the highest estimated EU wins. Implemented as `decisionpy.inference.numpyro.solvers.intervention_scan.solve`. The full solver family — the scan, the batched scan, and the backward-induction / SPU path — and the papers behind them are in [`solver_algorithms.md`](../inference/solver_algorithms.md).

### Trade-offs

|                     | A. Policy as params            | B. Intervention scan             |
| ------------------- | ------------------------------ | -------------------------------- |
| Best for            | continuous decisions, large action spaces | discrete decisions, small action spaces |
| Optimizes via       | gradients (fast, scales)       | enumeration / search (exact-ish) |
| Main difficulty     | gradient of `E[U]` w.r.t. policy — SVI's ELBO framing does not directly fit EU maximization, so a custom loss around `jax.grad` + trace is required | exponential blow-up in (#decisions × action-space) |
| Verifiability       | hard to tell convergence from correctness | trivially correct — EU measured per action |

The exact graph-native alternative to both — **bucket elimination** ([`bucket_elim.md`](../inference/bucket_elim.md), historical) — was the v0 solver: the factor machinery of variable elimination applied to the decision objective, exact optimal policies for all-categorical diagrams, numpy-only. It was retired together with the other exact engines (see `docs/ADR/25_08_2026_numpyro_only_engine.md`); its history lives in git.

---

## Status: Strategy B implemented; bucket elimination retired

The v0 solver was bucket elimination, chosen for the same criteria that motivated Strategy B: verifiability by hand for small diagrams and cross-validation against pyAgrum's exact LIMID solver, while sidestepping the genuinely hard part of Strategy A (the expected-utility gradient). It was exact, but restricted to all-categorical diagrams.

The NumPyro intervention scan has since replaced it, keeping the "EU measured per action" verifiability with none of that restriction:

- `solve()` handles mixed and continuous diagrams — the step the docs had always framed as Strategy B's purpose;
- the implementation is cross-validated against pyAgrum's exact LIMID solver (`tests/inference/numpyro/test_compare_pyagrum_limid.py`) and, for continuous posteriors, against `pyagrum.clg`;
- its scope is discrete decisions with discrete information sets, and its cost is the exponential policy space from the trade-off table above.

Strategy A (policy-as-parameters) remains the deferred path: it is what continuous decisions and continuous information sets will require.

Crucially, whichever solver runs, it commits only to the *algorithm*, not to the *node abstraction*. A `DecisionNode` declares "I have these possible actions / this support, and these parents I can observe." Whether the solver scans interventions or optimizes a policy is a solve-time choice.

Risks: the intervention scan is exponential in the policy space — it suits discrete decisions with small information sets. A continuous decision ("how much to invest") requires Strategy A, still deferred.

---

## Mixed discrete + continuous decisions

Mixed decisions are not an edge case — they are a first-class situation that classical influence-diagram tooling handles badly and a PPL backend handles well. They do not require a third strategy; they **nest**.

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
