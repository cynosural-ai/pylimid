# Policy families and mixed information

PyLIMID optimizes categorical and scalar continuous decisions through policy gradients. Chance variables and decision information can be categorical, continuous, or numerical counts. The creator chooses a family for every decision; the optimizer fits the joint parameters without changing the action spaces or information sets.

## Implemented families

`LinearSoftmax` computes linear scores from flattened numerical values and one-hot categorical features. The encoding is additive across parents and does not create categorical interaction features. Numerical inputs are not automatically scaled.

`TabularSoftmax` stores independent logits for every scalar categorical parent assignment. Its leading parameter axes follow the parent-state order and its last axis follows the action labels. A parentless decision has one action-score vector. The number of rows grows as the product of parent cardinalities, and rarely visited rows can receive little training.

Both families return the same fitted `CategoricalPolicy` representation. Each rule retains its family, a potentially nested parameter dictionary, information metadata, and action labels. The common wrapper exposes action probabilities, sampling with an explicit random key, and action log probabilities. The solver returns the final stochastic policies; taking an argmax creates a different deterministic policy and requires separate evaluation.

`DistributionPolicy` composes a scalar continuous distribution and a `ParameterModel`, with `Linear` as the built-in model. The fitted `ContinuousPolicy` exposes the conditional distribution, sampling, and log densities, and checks fixed scalar support against the action domain. Categorical and continuous rules can be trained together. The [decision-node note](../graph/decision_node.md) describes composition and execution.

## Binding and execution

Initialization walks the graph under the initial policies to determine information shapes. The built-in softmax families initialize to zero logits and uniform action probabilities. `Linear` initializes raw distribution-parameter predictions to zero, then `DistributionPolicy` applies the parameter constraints. For a Normal this gives location zero and scale one unless a parameter is fixed. Each family receives an explicit initialization key, and custom families can use it to break parameter symmetry. Every later training batch simulates complete scenarios under the current policies. Earlier actions can change downstream observations. Each decision sees only its declared parents, including remembered actions or observations.

The [optimizer design](../solver/algorithms.md) describes score gradients, the previous-batch baseline, and independent evaluation. The public call is `solve(diagram, num_steps=..., num_samples=..., learning_rate=..., rng_key=...)`; policy gradients are the only optimization engine.

Inference also accepts supplied deterministic functions and categorical action tables. Fitted categorical and continuous stochastic rules support forward evaluation and conditioning on chance evidence. Supplied deterministic functions support forward evaluation only.

## Verification and interpretation

Analytic gradients, exact categorical action sums, and continuous quadrature cover one or several decisions, zero and nonzero parameters, categorical outcomes of continuous actions, and action-dependent observations. End-to-end tests check categorical interactions, numerical information, explicit memory, mixed action types, bounded actions, and stochastic reuse. Posterior tests compare fitted continuous and mixed policies with analytic references and numerical integration. The self-contained [oil regression tests](../../tests/solver/test_oil_policy.py) sum the fitted policy's utility over a small categorical model and compare that value with the analytic optimum.

Three errors must be distinguished: approximation imposed by the family, incomplete optimization, and Monte Carlo evaluation noise. More evaluation draws address only the last. The optimizer reports no convergence or optimality certificate.

## Custom families

Custom families subclass `PolicyFamily` and implement `initialize` and `distribution`. Initialization returns a dictionary containing trainable JAX arrays, including nested parameter structures. Distribution construction receives those parameters and one information assignment in declared parent order, and returns a scalar-event NumPyro action distribution without batch dimensions. Categorical decisions require categorical probabilities or logits in declared action order. Continuous decisions require a continuous density with fixed support contained in their domain. Both methods must be pure and JAX-compatible, and log probabilities or densities must be differentiable with respect to trainable parameters. The fitted wrapper binds information metadata, checks distribution compatibility, and delegates sampling and scoring. Stochastic network layers and mutable network state need additional contracts.

Custom `ParameterModel` implementations can be composed with compatible distributions through `DistributionPolicy`.
