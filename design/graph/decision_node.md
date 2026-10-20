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

[Discrete decisions with continuous information](../continuous_information/README.md) are proposed separately from continuous actions. The action `states` and information-arc semantics stay the same. The creator additionally declares each decision's policy family, and the solver fits its parameters. Tables and functions share one policy abstraction. This extension can be implemented first; the exact family-declaration API remains open.

---

## What "solving a LIMID" means here

A LIMID is solved when each decision node has a local policy δᵢ, mapping its observed information to an action, and the policies together maximize expected total utility. Each policy uses only its declared information parents. Earlier observations and decisions are available only through explicit information arcs; the policies' effects on utility can still interact.

The objective is:

```
maximize   E[ Σ_k  U_k(parents(U_k)) ]
 over δ₁..δₙ
```

where the expectation is over chance variables, whose distributions depend on the chosen decisions through the graph.

---

## Two strategies

Both strategies use the same action spaces and information-arc semantics. In the proposed parameterized approach, the creator also declares the policy family to fit for each decision.

### Strategy A — policy as parameters (deferred)

The creator chooses the form of each decision's policy: for example, linear scores followed by softmax, or a neural network followed by softmax. The solver fits weights and biases to maximize expected utility within that family and returns the fitted rule. Choosing a family does not mean supplying the final strategy. A complete function supplied by the creator can instead be evaluated without optimization.

Solving uses an explicit expected-utility objective with an appropriate estimator: score-function gradients for sampled discrete actions or probability-weighted sums for enumerated actions. Continuous-action policies and their estimators remain a separate extension. NumPyro supplies probabilistic machinery; ordinary SVI or a utility `numpyro.factor()` alone does not establish an equivalent maximum-expected-utility objective. The [continuous-information proposal](../continuous_information/README.md#a-general-approximate-solver) develops the finite-action case.

### Strategy B — intervention scan

Decisions are discrete action choices. The solver enumerates the policy space — one action per information-set assignment per decision — and estimates each policy's expected utility by forward-sampling the diagram with every decision resolved deterministically from its observed information set (a Monte-Carlo estimate per policy). The policy with the highest estimated EU wins. Implemented as `pylimid.solver.scan.intervention.solve`; `solve(method="scan")` uses the batched implementation in `pylimid.solver.scan.batched`. The full solver family — the scan, the batched scan, and the backward-induction / SPU path — and the papers behind them are in [solver algorithms](../solver/algorithms.md).

### Trade-offs

|                     | A. Policy as params            | B. Intervention scan             |
| ------------------- | ------------------------------ | -------------------------------- |
| Best for            | discrete decisions with continuous information; future continuous actions | finite actions and categorical information sets |
| Optimizes via       | estimated policy gradients    | exhaustive policy search with sampled utility |
| Main difficulty     | gradient of `E[U]` w.r.t. policy — SVI's ELBO framing does not directly fit EU maximization, so a custom loss around `jax.grad` + trace is required | exponential blow-up in (#decisions × action-space) |
| Verifiability       | analytic examples and fresh policy evaluation; no general global guarantee | exhaustive search finds the best of the sampled policy estimates |

Bucket elimination was the original exact solver and was retired with the other exact engines; the [NumPyro-only ADR](../ADR/25_08_2026_numpyro_only_engine.md) records that decision. Solver choice does not change the graph abstraction: a `DecisionNode` declares its actions and the parents it can observe.

---

## Mixed discrete + continuous decisions

Continuous action domains remain future work, separate from allowing discrete decisions to observe continuous parents. Mixed action domains would require compatible policy families and estimators for both types of action.

When the discrete policy space is small and enumerable, one possible approach is to enumerate those policies and optimize the continuous policies within each branch. A discrete policy over continuous observations cannot generally be enumerated this way. Joint parameter optimization is another possible approach. Neither method is implemented, and non-convex optimization would need its own validation and guarantee analysis.

---

## Design implication

A decision's action domain, information set, and creator-selected policy family determine which solvers can handle it. A discrete action domain alone does not imply policy enumeration: a discrete decision with continuous observations can use a parameterized policy and gradient optimization.

The policy abstraction should support tables and function rules through a shared evaluation interface. Solvers fit compatible families and return reusable fitted policies. Continuous actions would extend this infrastructure with action domains and suitable optimization methods; they do not require a separate policy system.
