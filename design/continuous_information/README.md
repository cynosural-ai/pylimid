# Policy families and mixed information

PyLIMID optimizes finite-action influence diagrams through policy gradients. Chance variables and decision information can be categorical, continuous, or numerical counts. The creator chooses a family for every decision; the optimizer fits the joint parameters without changing the action spaces or information sets.

## Implemented families

`LinearSoftmax` computes linear scores from flattened numerical values and one-hot categorical features. The encoding is additive across parents and does not create categorical interaction features. Numerical inputs are not automatically scaled.

`TabularSoftmax` stores independent logits for every scalar categorical parent assignment. Its leading parameter axes follow the parent-state order and its last axis follows the action labels. A parentless decision has one action-score vector. The number of rows grows as the product of parent cardinalities, and rarely visited rows can receive little training.

Each fitted rule exposes action probabilities, sampling with an explicit random key, and action log probabilities. The solver returns the final stochastic policies; taking an argmax creates a different deterministic policy and requires separate evaluation.

## Binding and execution

Initialization walks the graph under uniform initial policies to determine information shapes. Every later training batch simulates complete scenarios under the current policies. Earlier actions can change downstream observations. Each decision sees only its declared parents, including remembered actions or observations.

The [optimizer design](../solver/algorithms.md) describes score gradients, the previous-batch baseline, and independent evaluation. The public call is `solve(diagram, num_steps=..., num_samples=..., learning_rate=..., rng_key=...)`; policy gradients are the only optimization engine.

Inference also accepts supplied deterministic functions and action tables. Fitted stochastic rules and supplied functions currently support forward evaluation without observed chance-node evidence. Conditioning through these rules needs a separate inference contract.

## Verification and interpretation

Analytic and exact-action gradient tests cover one or several decisions, zero and nonzero parameters, and action-dependent observations. End-to-end tests check categorical interactions, numerical information, explicit memory, mixed families, and stochastic reuse. The [oil example](../../examples/oil_policy.py) sums its fitted policy's utility over the small categorical model and compares that value with the analytic optimum.

Three errors must be distinguished: approximation imposed by the family, incomplete optimization, and Monte Carlo evaluation noise. More evaluation draws address only the last. The optimizer reports no convergence or optimality certificate.

## Future extensions

A custom-family interface should separate trainable parameters and initialization from fitted-rule execution. User-defined neural networks could supply action logits, with the categorical policy wrapper owning normalization and sampling. Random initialization, any additional network randomness, and mutable network state need explicit contracts.

Other future work includes richer baselines, checkpoint selection, optimizer diagnostics, and finite-action sums for variance reduction. Continuous actions require explicit domains and suitable policy distributions or other gradient estimators; they are separate from continuous observations.
