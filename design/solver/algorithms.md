# Policy-gradient optimization

PyLIMID uses joint policy gradients as its only policy-optimization engine. The graph can contain categorical, continuous, and numerical count chance variables. Decisions currently choose finite categorical actions. Each decision declares its information parents and a policy family; optimization preserves both.

## Objective and estimator

Let `theta` contain all decision-policy parameters. Complete simulated scenarios follow the chance model and sample actions from the current local policies. The objective is expected total utility:

```text
J(theta) = E_theta[sum of utility-node values]
```

For every sampled scenario, the score surrogate is:

```text
-(total_utility - baseline) * sum_i log pi_i(action_i | information_i)
```

Observations, sampled actions, utilities, and the baseline are held fixed when differentiating policy log probabilities. This accounts for earlier actions changing later observations without differentiating through categorical samples or requiring differentiable utility functions. Every function must still be traceable by JAX.

The baseline is the previous batch's mean utility; the first baseline is zero. Each Adam update uses a fresh batch. The solver returns the final iterate and estimates its utility on independent scenarios. A fixed training budget does not imply convergence or optimality.

## Families and local information

`LinearSoftmax` flattens numerical information and one-hot encodes categorical parents before computing linear action scores. `TabularSoftmax` stores an independent action-score row for each scalar categorical information assignment. Both produce stochastic categorical action distributions.

Families implement `PolicyFamily.initialize` and `PolicyFamily.logits`. Initialization receives an explicit random key and sampled information values; logits operate on one information assignment and a potentially nested parameter dictionary. The shared `CategoricalPolicy` binds the family, fitted parameters, information metadata, and action labels, and owns normalization, sampling, and log probabilities. The optimizer and inference backend use this common rule without dispatching on the family.

The graph controls information access. A rule receives only its declared parents; remembering an earlier observation or action requires an explicit information arc. Several families can be trained jointly, including decisions whose earlier actions change later observations.

`solve(diagram, num_steps=..., num_samples=..., learning_rate=..., rng_key=...)` is the public entry point and contains the training orchestration and private helpers. There is no solver selector or structural solution-ordering gate. `Solution` contains only the fitted joint policy and independently estimated expected utility. Its rules can be inspected, sampled, and reused through forward `infer` calls. `Solution.render()` uses the labels stored in its fitted rules, without requiring the original diagram.

Invalid solver inputs raise `ValueError`, including a diagram without decisions or a missing policy family. Non-finite fitted parameters or non-finite expected utility in the independent evaluation raise `SolveError`. The public errors remain in `pylimid.errors`; `InferenceError` covers inference requests and supplied-policy binding.

## Verification

Tests compare score gradients with analytic derivatives and exact action sums on small examples, including nonzero parameters and action-dependent observations. End-to-end tests cover mixed families, categorical interactions, explicit memory, count outcomes, reproducibility, and independent final evaluation. The oil example compares the exact value of its fitted policy with an analytic optimum; no reference solver is required.

General graph validation and distribution-contract tests remain separate from optimizer verification. Posterior inference retains analytic reference checks. The test suite has no pyAgrum dependency.

## Limits and future work

Training is approximate. Errors can arise from restricted policy families, imperfect optimization, and Monte Carlo evaluation. A utility confidence interval does not bound distance from the global optimum. Richer baselines, convergence diagnostics, checkpoint selection, custom neural families, and continuous action domains remain future work.

The score-function estimator follows [Schulman et al., Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254).
