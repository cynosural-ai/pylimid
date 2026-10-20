# Solver algorithms: the scan, the batched scan, and backward induction

This note collects the influence-diagram solving approaches — implemented and planned — and the papers behind them. It is the theory companion to `pylimid.solver` with estimator details in [backward induction](backward_induction.md).

## The problem

Solving an influence diagram means finding the decision rules — one action per information-set assignment per decision — that maximize the expected total utility, and reporting that optimum. pylimid diagrams are LIMIDs by construction: a decision's information set is exactly its drawn parents, with no implicit memory (see [decision nodes](../graph/decision_node.md)). The scan and the backward-induction path below are both Monte-Carlo solvers: they never build closed-form posteriors, which is what lets them handle mixed and continuous diagrams.

## Tier 1 — intervention scan (implemented)

The current solver enumerates the joint policy space and forward-samples every policy:

1. For each decision, build every information-set assignment (the product over its info parents' state counts).
2. For each decision, build every rule — one action per assignment — so a decision with K actions and |A| assignments has K^|A| rules.
3. The policy space is the product of the per-decision rule spaces: `∏_d K_d^|A_d|` policies.
4. Evaluate each policy by forward sampling (num_samples draws with every decision resolved from its info set), keep the best estimated EU.

Cost: `num_samples × ∏_d K_d^|A_d|`, with `|A_d| = ∏ over the info parents of their state counts`. This is doubly exponential in the number of info parents (the rule exponent is itself a product). It suits small decision spaces; continuous info parents are rejected because |A_d| would be infinite.

Guarantee: the global optimum of the Monte-Carlo estimate — every policy is evaluated, so the scan is the reference answer for the other solvers. Implementation: `pylimid.solver.scan.intervention.solve`.

## Tier 2 — batched scan (implemented)

The same algorithm with the same semantics and the same guarantees, engineered to be fast: all policies evaluated in one vmapped call with the policy arrays as a leading batch dimension, and the forward step jitted once — one trace, one dispatch for the whole solve. Because it uses the same PRNG key schedule per policy, it returns the same policy and the same EU as Tier 1, which makes the validation trivial: the two must agree (to float rounding, since the unbatched scan accumulates in float64 in Python).

The cost stays exponential (the enumeration is the same); what goes away is the per-policy JAX re-trace overhead, which dominated the runtime (measured ~260ms per 500-sample evaluation, ~60% of it trace machinery from user `dist` callables). Measured ~220× on a six-node soluble LIMID example: 112s → 0.5s at 2000 samples per policy. One constraint: utility `values` callables must be JAX-traceable (no `float()`/`int()` coercion), because they are evaluated inside the vmapped walk. Implementation: `pylimid.solver.scan.batched`.

## Tier 3 — backward induction / single policy updating (soluble diagrams implemented)

The estimator design — what is sampled, what is grouped, variance and validation — is pinned down in [`backward_induction.md`](./backward_induction.md).

The efficient alternative: solve one decision at a time, from the last to the first.

The principle is dynamic programming: consider the last decision. For each info-set assignment and each action, estimate the continuation value — the expected utility of everything that follows, given that assignment and action — by forward sampling and grouping trajectories by their realized information-set assignment. Pick the best action per assignment; the decision is resolved. Fold the resolved decision back into the problem and repeat for the previous decision. The key fact that makes this legal is that a later decision's optimal rule depends only on its information set, not on which actions earlier decisions will choose, so it can be solved before they are.

Cost: `Σ_d |A_d| × K_d` value estimates plus one argmax per decision — additive in the decisions, never the policy-product exponential.

Guarantees split along the regularity line:

- **Soluble diagrams (the gate in `pylimid/solver/regularity.py`):** an exact solution ordering exists — each decision can be finalized from its information set once the later decisions are resolved — so a single reverse-order pass is globally optimal. This generalizes Shachter's backward induction beyond no-forgetting. Exact in the potential-world, and still global when the values are Monte-Carlo estimates up to sampling noise.
- **True LIMIDs:** decisions may not remember relevant past information, so there is no clean order and a single pass is not valid. The fix is single policy updating (SPU): hold all rules fixed, improve one decision's rule given the others, sweep until no rule changes. This is coordinate ascent: monotone in expected utility, converges to a fixed point, but only a *local* optimum — the scan remains the ground truth when it matters.

Adding memory arcs (from an earlier decision or its observations to a later one) converts a true LIMID into a regular one — the classical no-forgetting assumption made explicit — at the price of larger information sets. It is a modeling change, not a solver trick: more memory can only weakly improve the achievable expected utility, but it is a different decision problem.

### pyAgrum's LIMID solver (the external reference)

Learned by testing pyAgrum against the fixtures: `ShaferShenoyLIMIDInference` implements the exact/general split as follows.

- It *is* backward induction, implemented as Shafer-Shenoy message passing in reverse decision order (the API exposes `reversePartialOrder`, `junctionTree`, `optimalDecision`, `MEU`), exact in one pass.
- It only solves regular ("solvable") LIMIDs: `isSolvable()` is the gate, and a non-regular diagram makes `makeInference()` raise "This LIMID/Influence Diagram is not solvable.".
- It has no SPU iteration — there is no `maxIteration` knob or iterative fallback. The remedy it offers is `addNoForgettingAssumption([...])`: add the memory arcs and change the model, then solve exactly.

This matters for validation: for solvable diagrams pyAgrum gives the exact reference, but for true LIMIDs there is no external exact solver to compare against — pyAgrum refuses them, so the scan is the only ground truth.

### The solvability gate: the paper vs aGrUM vs pylimid

One-pass backward induction is sound exactly on *soluble* LIMIDs, so the gate is part of the algorithm, not an optimization.

- **The paper (what we implement).** A decision is *extremal* when the utilities it can influence are d-separated from the families of *every* other unresolved decision by its own information set (Lauritzen and Nilsson 2001, Definition 11). A LIMID is *soluble* when an ordering exists in which each decision is extremal once the later ones are resolved (Definition 14), and the paper's algorithm searches for an extremal decision over the whole remaining set. `pylimid/solver/regularity.py` implements that search directly: repeatedly take any decision passing the global test; if none does, the diagram is not soluble.
- **aGrUM (the reference, with a hole).** `ShaferShenoyLIMIDInference` runs the same idea, but its `isSolvable()` compares candidates only against decisions sharing a level of an internal partial order (`_checkingSolvability_` in `ShaferShenoyLIMIDInference_tpl.h`). Decisions on different levels are never checked against each other, and the leveling itself is not in the published paper.
- **What the hole costs.** `D1 -> X -> D2`, both decisions feeding one utility `U = [[5, 4], [0, 3]]`, with `X` uninformative about `D1`: pyAgrum calls it solvable and returns MEU 4.0, while the feasible policy "both play a" scores 5.0. The paper's criterion rejects the diagram; our gate raises and the scan finds 5.0. The regression is pinned in `tests/solver/test_regularity.py` and `test_backward_induction.py`.
- **The divergence is deliberate.** `is_solvable` matches `isSolvable()` on every fixture and structural battery case except the pinned one, where the paper and pyAgrum disagree. A diagram pyAgrum solves that we reject is not a false alarm — it is a diagram on which pyAgrum can silently return a suboptimal policy.
- **Not implemented from the paper:** the minimal-reduction loop (repeated removal of non-requisite arcs to test the reduced model). That loop explores reduced models and bounds; the correctness gate for solving the diagram as given is the exact-solution ordering above.

## Discrete decisions with continuous information (single decision implemented)

The table tiers above all require categorical information sets. Supporting a continuous observation does not require a continuous action: a finite-action decision can instead use a function policy. In the [continuous-information proposal](../continuous_information/README.md), the diagram creator chooses a policy family per decision, such as linear scores followed by softmax or a neural network followed by softmax. A compatible solver fits its parameters and returns the fitted policy. Tables and functions share one policy abstraction.

`solve(method="policy_gradient")` now implements sampled-action score gradients and Adam for exactly one decision declaring `policy_family=LinearSoftmax()`. It returns a fitted stochastic rule and evaluates its utility on fresh scenarios. It preserves the declared information set and family, but has no global or local optimality guarantee, even within that family. General multiple-decision training and finite action enumeration for variance reduction remain future work. Continuous actions remain separate.

## The papers

- **Bellman, R. (1957). Dynamic Programming. Princeton University Press.** — the origin of the backward-induction principle; decision trees have been solved this way since long before influence diagrams existed.
- **Howard, R. A. and Matheson, J. E. (1984). Influence diagrams. In Readings on the Principles and Applications of Decision Analysis, Vol. II, 719–762.** — introduced influence diagrams, with full memory (no-forgetting) assumed.
- **Shachter, R. D. (1986). Evaluating influence diagrams. Operations Research 34(6), 871–882.** — the classical algorithm: backward induction / arc reversals on regular IDs.
- **Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science 47(9), 1235–1251.** — the original LIMID paper: drops the no-forgetting assumption, defines soluble (regular) vs. non-soluble LIMIDs, and proposes single policy updating (SPU) for the general case; backward induction is the special case that finishes in one pass. This is the closest reference for the whole solver family, and the source of the extremality criterion the gate implements.
- **Bielza, C., Müller, P. and Ríos Insua, D. (1999). Decision analysis by augmented probability simulation. Management Science 45(7), 995–1007.** — Monte-Carlo methods for solving decision problems; the tradition the scan and the sampling-based backward induction belong to (the papers above are exact/potential-based).
- **Kearns, M., Mansour, Y. and Ng, A. (1999). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. IJCAI'99.** — sampling-based dynamic programming in the MDP world; the same "estimate continuation values by simulation, then argmax" idea that Level 2 applies to influence diagrams.

## How each tier is validated

- **Batched scan:** must reproduce the scan bit-for-bit (same PRNG key schedule) — a pure-performance change with no behavior change.
- **Backward induction (soluble diagrams):** compare against pyAgrum's exact LIMID solver (Shafer-Shenoy message passing) at Monte-Carlo tolerances — a six-node soluble LIMID is a natural fixture, and `isSolvable()` must be true for it. The gate itself follows the paper's criterion, which is stricter than pyAgrum's on the pinned divergence; there pyAgrum is the wrong reference and the scan is the only one.
- **SPU (true LIMIDs):** compare against the scan at Monte-Carlo tolerances — pyAgrum refuses non-regular LIMIDs outright (it has no SPU), so the scan is the only global reference for that case.

## Status

- Tier 1: implemented (`pylimid.solver.scan.intervention`).
- Tier 2: implemented (`pylimid.solver.scan.batched`).
- Tier 3: implemented for soluble diagrams (`pylimid.solver.backward_induction`), gated by `pylimid.solver.regularity` with the paper's exact-solution-ordering criterion (stricter than pyAgrum's level-based `isSolvable`, see above) and validated against pyAgrum and the scan; SPU for non-soluble LIMIDs deferred.
