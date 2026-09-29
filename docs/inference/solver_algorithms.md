# Solver algorithms: the scan, the batched scan, and backward induction

This note collects the influence-diagram solving approaches — implemented and planned — and the papers behind them. It is the theory companion to `decisionpy.inference.numpyro.solvers` and to the roadmap in REMAINING_WORK.md (items 4 and 5).

## The problem

Solving an influence diagram means finding the decision rules — one action per information-set assignment per decision — that maximize the expected total utility, and reporting that optimum. decisionpy diagrams are LIMIDs by construction: a decision's information set is exactly its drawn parents, with no implicit memory (see decision_node.md). The scan and the backward-induction path below are both Monte-Carlo solvers: they never build closed-form posteriors, which is what lets them handle mixed and continuous diagrams.

## Tier 1 — intervention scan (implemented)

The current solver enumerates the joint policy space and forward-samples every policy:

1. For each decision, build every information-set assignment (the product over its info parents' state counts).
2. For each decision, build every rule — one action per assignment — so a decision with K actions and |A| assignments has K^|A| rules.
3. The policy space is the product of the per-decision rule spaces: `∏_d K_d^|A_d|` policies.
4. Evaluate each policy by forward sampling (num_samples draws with every decision resolved from its info set), keep the best estimated EU.

Cost: `num_samples × ∏_d K_d^|A_d|`, with `|A_d| = ∏ over the info parents of their state counts`. This is doubly exponential in the number of info parents (the rule exponent is itself a product). It suits small decision spaces; continuous info parents are rejected because |A_d| would be infinite.

Guarantee: the global optimum of the Monte-Carlo estimate — every policy is evaluated, so the scan is the reference answer for the other solvers. Implementation: `decisionpy.inference.numpyro.solvers.intervention_scan.solve`.

## Tier 2 — batched scan (implemented)

The same algorithm with the same semantics and the same guarantees, engineered to be fast: all policies evaluated in one vmapped call with the policy arrays as a leading batch dimension, and the forward step jitted once — one trace, one dispatch for the whole solve. Because it uses the same PRNG key schedule per policy, it returns the same policy and the same EU as Tier 1, which makes the validation trivial: the two must agree (to float rounding, since the unbatched scan accumulates in float64 in Python).

The cost stays exponential (the enumeration is the same); what goes away is the per-policy JAX re-trace overhead, which dominated the runtime (measured ~260ms per 500-sample evaluation, ~60% of it trace machinery from user `dist` callables). Measured ~220× on the Oil Wildcatter fixture: 112s → 0.5s at 2000 samples per policy. One constraint: utility `values` callables must be JAX-traceable (no `float()`/`int()` coercion), because they are evaluated inside the vmapped walk. Implementation: `decisionpy.inference.numpyro.solvers.batched_scan`.

## Tier 3 — backward induction / single policy updating (regular diagrams implemented)

The estimator design — what is sampled, what is grouped, variance and validation — is pinned down in [`backward_induction.md`](./backward_induction.md).

The efficient alternative: solve one decision at a time, from the last to the first.

The principle is dynamic programming: consider the last decision. For each info-set assignment and each action, estimate the continuation value — the expected utility of everything that follows, given that assignment and action — by forward sampling (each assignment gets its own dedicated samples: stratified). Pick the best action per assignment; the decision is resolved. Fold the resolved decision back into the problem and repeat for the previous decision. The key fact that makes this legal is that a later decision's optimal rule depends only on its information set, not on which actions earlier decisions will choose, so it can be solved before they are.

Cost: `Σ_d |A_d| × K_d` value estimates plus one argmax per decision — additive in the decisions, never the policy-product exponential.

Guarantees split along the regularity line:

- **Regular diagrams (no-forgetting):** the info sets are closed under an ordering of the decisions, so a single reverse-order pass is globally optimal — this is Shachter's backward induction. Exact in the potential-world, and still global when the values are Monte-Carlo estimates.
- **True LIMIDs:** decisions may not remember relevant past information, so there is no clean order and a single pass is not valid. The fix is single policy updating (SPU): hold all rules fixed, improve one decision's rule given the others, sweep until no rule changes. This is coordinate ascent: monotone in expected utility, converges to a fixed point, but only a *local* optimum — the scan remains the ground truth when it matters.

Adding memory arcs (from an earlier decision or its observations to a later one) converts a true LIMID into a regular one — the classical no-forgetting assumption made explicit — at the price of larger information sets. It is a modeling change, not a solver trick: more memory can only weakly improve the achievable expected utility, but it is a different decision problem.

### pyAgrum's LIMID solver (the external reference)

Learned by testing pyAgrum against the fixtures: `ShaferShenoyLIMIDInference` implements the exact/general split as follows.

- It *is* backward induction, implemented as Shafer-Shenoy message passing in reverse decision order (the API exposes `reversePartialOrder`, `junctionTree`, `optimalDecision`, `MEU`), exact in one pass.
- It only solves regular ("solvable") LIMIDs: `isSolvable()` is the gate, and a non-regular diagram makes `makeInference()` raise "This LIMID/Influence Diagram is not solvable.".
- It has no SPU iteration — there is no `maxIteration` knob or iterative fallback. The remedy it offers is `addNoForgettingAssumption([...])`: add the memory arcs and change the model, then solve exactly.
- The planned `is_solvable` gate in our Level 2 solver mirrors this check; the difference is that our values are Monte-Carlo estimates, so a regular diagram still needs the usual MC tolerances.

This matters for validation: for regular diagrams pyAgrum gives the exact reference, but for true LIMIDs there is no external exact solver to compare against — pyAgrum refuses them, so the scan is the only ground truth.

## The papers

- **Bellman, R. (1957). Dynamic Programming. Princeton University Press.** — the origin of the backward-induction principle; decision trees have been solved this way since long before influence diagrams existed.
- **Howard, R. A. and Matheson, J. E. (1984). Influence diagrams. In Readings on the Principles and Applications of Decision Analysis, Vol. II, 719–762.** — introduced influence diagrams, with full memory (no-forgetting) assumed.
- **Shachter, R. D. (1986). Evaluating influence diagrams. Operations Research 34(6), 871–882.** — the classical algorithm: backward induction / arc reversals on regular IDs.
- **Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science 47(9), 1235–1251.** — the original LIMID paper: drops the no-forgetting assumption, defines regular vs. non-regular LIMIDs, and proposes single policy updating (SPU) for the general case; backward induction is the special case that finishes in one pass. This is the closest reference for the whole solver family.
- **Bielza, C., Müller, P. and Ríos Insua, D. (2007). Decision analysis by augmented probability simulation. Management Science 53(7).** — Monte-Carlo methods for solving decision problems; the tradition the scan and the sampling-based backward induction belong to (the papers above are exact/potential-based).
- **Kearns, M., Mansour, Y. and Ng, A. (1999). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. IJCAI'99.** — sampling-based dynamic programming in the MDP world; the same "estimate continuation values by simulation, then argmax" idea that Level 2 applies to influence diagrams.

## How each tier is validated

- **Batched scan:** must reproduce the scan bit-for-bit (same PRNG key schedule) — a pure-performance change with no behavior change.
- **Backward induction (regular diagrams):** compare against pyAgrum's exact LIMID solver (Shafer-Shenoy message passing) at Monte-Carlo tolerances — the Oil Wildcatter is a natural fixture, and `isSolvable()` must be true for it.
- **SPU (true LIMIDs):** compare against the scan at Monte-Carlo tolerances — pyAgrum refuses non-regular LIMIDs outright (it has no SPU), so the scan is the only global reference for that case.

## Status

- Tier 1: implemented (`decisionpy.inference.numpyro.solvers.intervention_scan`).
- Tier 2: implemented (`decisionpy.inference.numpyro.solvers.batched_scan`).
- Tier 3: implemented for regular diagrams
  (`decisionpy.inference.numpyro.solvers.backward_induction`), gated by
  `...solvers.regularity` and validated against pyAgrum and the scan;
  SPU for non-regular LIMIDs planned — REMAINING_WORK item 5.
