# Backward induction (Level 2): the estimator design

Short design note for the planned per-decision solver — Tier 3 in [`solver_algorithms.md`](./solver_algorithms.md), REMAINING_WORK item 5. It pins down the Monte-Carlo estimator before implementation: what is sampled, what is grouped, how the decisions interact, and where the variance lives.

## Scope

- **Regular LIMIDs** (no-forgetting) solved in one reverse-order pass. Non-regular diagrams are gated out by `is_solvable` — the first version raises a clean error naming the scan as the alternative rather than silently falling back into an exponential solver; SPU comes later.
- **Discrete action spaces and discrete information sets.** Continuous information sets remain the deferred Strategy A.
- Values are Monte-Carlo estimates, like every solver in the family.

## What is estimated

For decision `d`, information-set assignment `a` and action `k`:

    Q(a, k) = E[total utility | info(d) = a, do(d = k)]

with every *later* decision resolved by its already-computed rule, and every *earlier* decision acting under a placeholder rule.

The later rules are fixed by the time `d` is solved, so the estimate depends on them — that is the point of the reverse order. The placeholder rules for earlier decisions must not matter, and under regularity they don't: `info(d) = a` determines every earlier decision's action (their information sets are contained in `d`'s), so the conditional expectation is invariant to how those actions were generated. That invariance is what makes the reverse pass sound.

## How it is estimated: stratified grouping

For each decision `d`, in reverse order, and each action `k`:

1. Sample `num_samples` forward trajectories with `d` clamped to `k`, later decisions resolved by their rules, earlier decisions under the placeholder rule (uniform over actions is the simple choice, and it covers the info assignments evenly).
2. Group the trajectories by the **realized information-set assignment** of `d` — the sampled values of its parents.
3. `Q̂(a, k)` = mean total utility within the group for assignment `a`.

Then `rule(a) = argmax_k Q̂(a, k)`; `d`'s rule is fixed; move to the previous decision.

Cost: `Σ_d K_d × num_samples` trajectories — additive in the decisions, versus the scan's `∏_d K_d^|A_d|` policies.

Batching follows the batched scan: the per-action passes are vmapped forward walks, and the group means are segment-means over assignment indices (bincount-style), so one jitted call can cover a decision's whole update. The traceability constraint on `values` callables from the batched scan applies unchanged.

## Variance: the careful part

- Rare information-set assignments produce small groups, so `Q̂` is noisy and the argmax can flip on contested assignments. The bias is local to a group, unlike the scan's max-over-policies selection bias.
- Mitigations, in order of effort: a placeholder policy that covers assignments evenly (uniform); enough samples per action; validation against pyAgrum (exact) on the fixtures, so a flip cannot pass silently. Explicit stratification over assignments — sampling each assignment's group directly — would need conditioning and is out of scope for now.

## Regularity: `is_solvable`

The deciding property: the decisions admit an ordering `d1 < ... < dn` such that each `di`'s information set contains all earlier decisions and their information sets. The check mirrors pyAgrum's `ShaferShenoyLIMIDInference.isSolvable()` — implement it and pin agreement on the LIMID fixtures and the Oil Wildcatter. Adding memory arcs turns a non-regular diagram into a regular one, but that is a modeling change the user makes (pyAgrum's `addNoForgettingAssumption`), not something the solver does.

## Validation plan

- `is_solvable` equals pyAgrum's `isSolvable()` on every fixture (regular and non-regular).
- Regular diagrams: policy and expected utility against pyAgrum's exact solver at Monte-Carlo tolerances (LIMID fixtures + Oil Wildcatter).
- Non-regular: the error path; when SPU lands, against the scan at MC tolerance — the scan is the only global reference there, since pyAgrum refuses non-regular LIMIDs.
- Determinism: a fixed `rng_key` reproduces the same policy.

## Deferred

- SPU iteration for non-regular LIMIDs — an outer loop re-applying the same per-decision update until no rule changes (local optimum).
- Continuous information sets (Strategy A).
- Explicit per-assignment stratification and smarter sampling policies.
