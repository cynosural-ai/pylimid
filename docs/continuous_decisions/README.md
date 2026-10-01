# Continuous decisions — scope exploration

**Status:** exploration, not a commitment. Nothing described here is implemented, and the current solver family and `DecisionNode` representation are unchanged. This note exists so the "how far do we reach?" decision is made against a written map instead of discovered mid-implementation. The roadmap placeholder remains REMAINING_WORK item 3 (Strategy A).

## Why this note

decisionpy today solves influence diagrams whose decisions are **discrete** — a declared action space — with **discrete information sets** (tabulated). Continuous *chance* nodes are already handled: the intervention scan and backward induction estimate expected utility for mixed/continuous diagrams by Monte Carlo. The missing half is decisions with real-valued actions ("how much to invest") and information sets containing real-valued observations ("the biomarker reading"); both are rejected by the graph layer and by every solver.

Strategy A (policy as parameters), described in [`decision_node.md`](../graph/decision_node.md), is the deferred path for those models. This note surveys the extension: what is mathematically and practically reachable, what each reach costs, and where the library's guarantees change. It is deliberately a map, not an ADR.

## The boundary today (code map)

- **Representation.** `DecisionNode.states = None` means *not-yet-configured*: `_compute_consistency` returns `UNCONFIGURED` and `InfluenceDiagram.validate()` reports it, so `snapshot()` raises. There is no way to declare a continuous decision that passes validation, and no action domain (support/bounds) is expressible (`decisionpy/graph/decision_node.py`, `decisionpy/graph/diagram.py`).
- **Policy machinery.** `info_assignments` rejects any continuous information parent; `policy_space` asserts `states is not None`; `policy_array` builds integer lookup tables indexed by parent-state assignments (`decisionpy/inference/numpyro/solvers/_policy.py`).
- **Solvers.** The scan, the batched scan, and backward induction resolve decisions by indexing integer tables; backward induction takes an `argmax` over discrete actions (`decisionpy/inference/numpyro/solvers/`).
- **Result contract.** `Policy = dict[str, dict[tuple[int, ...], int]]` — a policy is a table of integer actions. `infer(policy=...)` binds integer actions and `to_model` clamps a decision to a constant (`decisionpy/inference/result.py`, `engine.py`, `numpyro/model.py`).
- **Already in place.** Continuous chance nodes sample and infer today (enumeration for discrete, NUTS for continuous); `UtilityNode.values` is already an arbitrary callable; the forward walks are already vmapped and jitted (`numpyro/samplers.py`, `numpyro/model.py`, `solvers/batched_scan.py`).

## The shift: from tables to functions

A discrete solver searches over **policy tables**: one action per information-set assignment, a finite set, enumerated exhaustively. A continuous solver cannot enumerate: the policy space is every function from the information set to the action space. The practical move is to parameterize the policy — a small number of parameters per decision — and optimize those parameters to maximize expected utility.

| | Discrete action (today) | Continuous action |
| --- | --- | --- |
| Policy form | table: assignment -> action | function: information -> real value |
| Search | enumerate every table | optimize parameters by gradient |
| Answer | global best over the table space | stationary point / local optimum |
| Evaluation | Monte Carlo per candidate policy | Monte Carlo per gradient step |
| Cost | exponential in decisions x information assignments | depends on policy dimension and samples |

Two properties make the continuous case tractable at all: the utility is already a deterministic function of sampled values, and automatic differentiation can differentiate the *simulation* that produces those values (see [`solver_methods.md`](./solver_methods.md)). The method this points at is **deterministic policy optimization through a differentiable simulation**; "policy gradient" is reserved for the score-function variant that discrete selections later require (see [`solver_methods.md`](./solver_methods.md#terminology-and-naming)).

## The four axes that determine difficulty

Variable types alone do not decide how hard a diagram is. Four mostly independent axes do:

1. **Action selection** — is the policy a table or a function?
2. **Where discreteness sits relative to the decision** — on the decision's causal path to the utility, or off it?
3. **Utility regularity** — smooth, kinked, or discontinuous?
4. **Size** — number of decisions (nesting) and information-set dimension (curse of dimensionality).

### Axis 1 — action selection

| Action | Information set | Policy form | How it is optimized |
| --- | --- | --- | --- |
| discrete | discrete | table (assignment -> action) | enumeration — current, global, exponential |
| discrete | continuous | parameterized classifier (e.g. softmax) | score-function gradient, or discretize the observation |
| continuous | discrete | table of real numbers | pathwise gradients — easy variant |
| continuous | continuous | parameterized function (affine, MLP) | pathwise gradients |

Selecting a discrete action is non-differentiable. That is a property of the policy, independent of the chance nodes: a discrete decision with a continuous information set is a caveat even before any mixed chance node enters the picture.

### Axis 2 — where discreteness sits

The single most useful rule:

> Gradient methods differentiate the path from a decision's value to the utility. Discreteness **on that path** is the problem; discreteness **off the path** is harmless, because the solver either conditions on it (observed information) or averages over it (Monte Carlo).

- Discrete chance **upstream** of the decision (an information parent or an ancestor): harmless — the policy conditions on the sampled value.
- Discrete chance **in an uninfluenced branch**: harmless — averaged over by Monte Carlo.
- Discrete chance **downstream of a continuous decision** and on a path to the utility: the poison case. A sampled discrete value jumps as the action moves, so the pathwise gradient is zero almost everywhere. Fixes: exact enumeration of that node inside the objective (a smooth conditional expectation, at the cost of its state count) or a score-function estimator (unbiased, higher variance).
- Discrete chance **downstream of a discrete decision**: harmless — the action is chosen/enumerated, not differentiated through.
- The same up/downstream rule applies to non-reparameterizable continuous distributions: differentiable sampling matters, Gaussianity does not.

"Continuous node feeding a discrete node" matters only when the continuous node is itself influenced by the decision: `d -> x -> s -> U` with `s` discrete needs the discrete-node treatment; an independent `x -> s` does not.

### Axis 3 — utility regularity

Summarized [below](#utility-regularity-levels), with examples in [`solver_methods.md`](./solver_methods.md).

### Axis 4 — size

- **Number of decisions.** Multiple decisions compose. Discrete+discrete enumerates a product; continuous+continuous optimizes jointly or in reverse order; mixed nests enumeration around inner optimization. LIMID structure (local information sets) keeps this from being worse, but there is no free lunch.
- **Information dimension.** A continuous information set of dimension `k` cannot be tabulated; the policy must generalize over `R^k`. Affine policies scale gracefully; flexible ones hit the curse of dimensionality and sample cost.

## Model classes, easiest to hardest

**1. All continuous (chance and decisions).** Feasible. The enabling condition is not "Gaussian" but **reparameterizable** differentiable sampling: Normal, Gamma, Beta, Dirichlet, StudentT, and transformed distributions qualify (NumPyro provides implicit reparameterization for many of them). Caveats are the universal ones: local optima, Monte-Carlo noise, policy-class choice, information dimension, and utility regularity. Gaussianity itself buys only the LQG closed form, which is useful as a test oracle (see [`literature.md`](./literature.md)).

**2. Continuous decisions, mixed chance nodes.** Feasible with a topology-dependent caveat. The question is not whether discrete chance nodes exist, but whether the decision can move one that reaches the utility (Axis 2). Discrete nodes upstream of, or unreachable from, the decision are harmless; a discrete node downstream of the decision needs exact enumeration or a score-function estimator.

**3. Mixed decisions, continuous chance nodes.** Feasible with caveats that concern action selection and nesting. A discrete decision with continuous information cannot be tabulated; it needs a parameterized classifier and a non-pathwise estimator, or a discretized observation. Mixing decision types composes the two solver styles: enumerate the discrete branches and optimize continuous policies inside, or jointly optimize all policies with the discrete selections handled by score/enumeration. Cost multiplies; there is no global guarantee.

**4. Mixed decisions, mixed chance nodes.** The hardest. It is the union of 2 and 3 plus their interactions (discrete chance descendants of continuous decisions, discrete decisions with continuous information, nesting). Feasible in principle, with the largest engineering and variance burden.

| Class | Main caveat |
| --- | --- |
| all continuous | local optima, MC noise, reparameterizability, policy class |
| continuous decisions + mixed chance | discrete descendants of the decision -> score function or enumeration |
| mixed decisions + continuous chance | discrete decisions with continuous info -> score/classifier; nesting cost |
| mixed decisions + mixed chance | union of the above |

## Utility regularity levels

Utility is always a deterministic function of its parents — that never changes with the model class. What changes is how the function, and the sampled estimate of its expectation, behave.

| Level | Shape | Example | Consequence |
| --- | --- | --- | --- |
| **Smooth** | polynomial, exp, log, sigmoid | `-(T - 20)^2 - a^2` | ideal; pathwise gradients, low variance. LQG lives here |
| **Kinked** | min/max/ReLU | newsvendor `p*min(D, a) - c*a` | the per-sample derivative is a subgradient; the expectation is usually still smooth (the kink moves with the randomness); noisy but workable |
| **Discontinuous** | indicators, thresholds, rounding | bonus `1{a >= D}` | the naive pathwise gradient is zero almost everywhere — unusable as-is |

The last row holds a subtle lesson: **a smooth objective can still have a useless estimator**. `E[1{D <= a}] = P(D <= a)` is perfectly smooth in `a` — it is the CDF — but the sampled indicator has no gradient. Gradient methods need a usable estimator, not merely a smooth objective; the fixes are an analytic expectation (CDF), a sigmoid relaxation, the score function, or enumeration.

## What is feasible

**Comfortably feasible.** Continuous decisions with reparameterizable chance nodes and smooth (or mildly kinked) utility; discrete chance nodes off the differentiated path; continuous information sets handled by affine policies. This is where LQG lives and where the analytic oracle exists.

**Feasible with caveats.** Discrete chance descendants of a continuous decision (enumeration or score function); non-reparameterizable continuous distributions (implicit reparameterization where available, else score function); discrete decisions with continuous information (classifier plus score function); mixed discrete/continuous decision nesting (cost and convergence); discontinuous utility at the sample level (reformulate to a usable estimator).

**To give up in general.** Global optimality for continuous decisions; exact answers; cheap treatment of genuinely non-differentiable or non-reparameterizable structure; high-dimensional information at low sample cost.

## The guarantee bargain

The current discrete solver enumerates every policy, so it can promise the **global** best policy of its Monte-Carlo estimate — exact search, approximate evaluation — and pays exponential cost for that promise. A continuous solver cannot enumerate an infinite policy space, so it can only promise a good local optimum — approximate search, approximate evaluation. This is the real design decision of this branch: whether the library is willing to change the meaning of `solve()` for continuous models, and how honestly the result communicates it. The existing trade-off table in [`decision_node.md`](../graph/decision_node.md) already flags the asymmetry: Strategy B is "trivially correct — EU measured per action", while Strategy A's difficulty is that convergence is hard to distinguish from correctness.

## Validating continuous-decision solvers

There is no external oracle: pyAgrum solves discrete LIMIDs, and its continuous engine (`pyagrum.clg`) is a Bayesian network only, with no decisions. The validation plan therefore rests on:

1. **Analytic LQG.** For linear-Gaussian models with quadratic utility the optimal policy and expected utility are closed-form (certainty equivalence / separation principle). The solver must recover the known coefficients. This is the strongest available check.
2. **Discretize and compare.** Bin the continuous action (and any continuous information) onto a grid, solve the discretized model with the existing scan, and check that the continuous solution approaches the grid optimum as the grid refines, within discretization bias.
3. **Monte-Carlo invariants.** Common random numbers, monotone improvement across optimizer steps, seed stability, and value-of-information comparisons (no information <= noisy information <= full information).

## Scope ladder: how far to reach

The branch's question is which rung to aim for.

| Reach | Unlocks | Requires | Risk / effort |
| --- | --- | --- | --- |
| **0 — today** | discrete decisions only | — | — |
| **1 — all-continuous differentiable** | real actions and observations; LQG; non-Gaussian reparameterizable chance; smooth and kinked utility | continuous-decision representation, parameterized policies, pathwise MC objective, result-contract change | low-moderate; the math is standard; roughly 1-2 weeks for an MVP, validated by LQG + grids |
| **2 — + discrete chance descendants** | continuous decisions in genuinely mixed-chance diagrams | exact conditional enumeration of discrete descendants (or a score function) in the differentiable walk | moderate; variance and engineering |
| **3 — + discrete decisions with continuous info, and nesting** | full mixed LIMIDs | classifier policies plus score function; outer enumeration / inner optimization composition | high; variance, convergence, cost |
| **4 — research** | non-reparameterizable distributions at scale, high-dimensional neural policies, global guarantees | beyond the current design | not planned; out of scope for a first implementation |

A sensible first milestone is **Reach 1 with loud rejection** of everything it cannot handle (discrete descendants on the differentiated path, discrete decisions with continuous information), so the supported class is explicit rather than silently wrong. Reach 2 is a natural follow-on because exact enumeration is a bounded, well-understood addition for the small discrete nodes typical of these models.

## When to revisit

This note is parked, not abandoned. It is a map for the eventual "how far do we reach?" decision, not a plan being executed. The trigger to pick the extension back up is that the discrete-decision work is finished:

- the discrete solvers are complete — SPU iteration landed and the `is_solvable` gate stopped being load-bearing;
- the example set covers the discrete decision classes the library is meant to demonstrate;
- the documentation and the code layout are settled, so a continuous extension lands on a stable surface instead of moving under it.

Until then, implementation-level detail is deliberately left to that point: constraint handling, convergence criteria and how the result reports its Monte-Carlo uncertainty, discrete-selection relaxations, and the policy/result API.

## Open design questions

- **How is a continuous decision declared?** Keep `states = None` and add a `support`/`bounds` field that makes the node CONSISTENT, or an explicit continuous marker. The action domain is needed for bounds/squashing and to distinguish "continuous" from "not yet configured".
- **How is a continuous policy represented in the result?** A `Solution.policy` entry for a continuous decision is a function, not a table. Options: return a callable; return parameters plus a documented family; or a discriminated union of table and function. It must be consumable by `infer(policy=...)`, which currently binds integers.
- **How does `solve()` dispatch, and what does it promise?** Discrete diagrams keep the exhaustive, global solver; continuous ones take the approximate path. The result may need to expose which kind of answer was produced.
- **Is a discretized fallback kept?** A grid over the action plus the existing scan is a small-problem reference and a validation tool; decide whether it is a public convenience or only a test technique.
- **Where do policy parameters and optimizer settings live?** Solver-owned defaults (affine, fixed steps) is a smaller API; user-supplied policy families are more expressive. Reach 1 likely starts solver-owned.

## Non-goals (parked)

- Global optimality and exact answers for continuous-decision diagrams.
- Learning from data, exploration, or online interaction — the model is known; this is planning, not reinforcement learning.
- Loops and long horizons — diagrams are acyclic; sequential decision making is expressed with memory arcs, already supported for discrete decisions.
- Neural policy training as a core feature; it can be an advanced policy family later.

## See also

- [`solver_methods.md`](./solver_methods.md) — how a Strategy A solver works: policy parameterization, pathwise/score/enumeration gradient estimators, the optimization loop, a worked thermostat example, and utility pathologies.
- [`utility_contract.md`](./utility_contract.md) — writing utilities for the gradient solver: traceable vs differentiable callables, the good/bad patterns, the silent-vs-loud failure modes, and the three enforcement layers.
- [`literature.md`](./literature.md) — annotated reading map: influence diagrams and LIMIDs, Monte-Carlo decision analysis, gradient estimation, control as inference, model-based policy search, and probabilistic-programming systems.
