# Discrete decisions with continuous information

**Status:** stages 1 and 2 are implemented: deterministic function evaluation, and explicit linear-softmax training for exactly one decision with stochastic forward reuse. Multiple-decision optimization and custom trainable families remain proposals. This is a separate extension from continuous actions. Every decision in scope still declares a finite tuple of action `states`; what changes is the policy's ability to use continuous observations.

The diagram creator chooses each decision's policy family: the form of the rule connecting its observations to its actions. The solver fits that family's parameters to maximize expected utility and returns the fitted policy. Tables and functions are supported representations under one policy abstraction; a neural network is optional.

## Scope and sequencing

An action space and an information set are different parts of a decision problem:

| Actions | Observations | Policy representation | Status |
| --- | --- | --- | --- |
| Discrete | Finite categorical states | Table from parent-state assignments to action indices | Implemented |
| Discrete | Continuous, or mixed categorical and continuous | Function from observed values to action indices or action probabilities | Deterministic and fitted stochastic forward evaluation implemented; linear-softmax training implemented for one decision |
| Continuous | Categorical, continuous, or mixed | Function returning a real-valued action | Separate continuous-action proposal |

Observing an exact temperature and choosing **fan on/off** belongs to this proposal. Observing temperature and choosing **heater power** requires continuous actions. Continuous chance variables elsewhere in the graph are already supported and do not by themselves require either extension.

The existing [continuous-decision draft](https://github.com/cynosural-ai/pylimid/pull/41) groups both missing capabilities into a wider exploration. This note separates the discrete-action case so its policy representation, optimizer, and validation can be reviewed and implemented without declaring continuous decisions, action bounds, or differentiable action-to-utility paths.

Stages 1 and 2 establish evaluation and single-decision training. Next extend training to several decisions, then revisit continuous actions. The two action types need not use the same gradient estimator.

## Influence-diagram semantics

An arc into a decision means that the parent value is observed before acting. A continuous observation does not change that meaning: the decision rule can be a threshold, a piecewise function, or another function of the observed value. Hybrid influence diagrams with continuous chance variables and discrete decisions are established in the literature; see the [references](#references).

A deterministic policy is an action-valued function `a = delta(h)`, where `h` contains exactly the decision's information parents. A randomized policy is a conditional distribution `pi(a | h)`. A softmax is one possible parameterization of that distribution, not a requirement imposed by the graph.

The policy parameters are chosen to maximize utility. They are not unknown chance-node parameters inferred from data. A decision keeps its controlled-choice semantics even if a solver samples its action during optimization.

Information constraints remain explicit. A policy may not inspect an unobserved ancestor, another node outside its information set, or remembered history unless the corresponding parents are drawn. Earlier decisions and observations are remembered only through those explicit arcs.

## Current implementation boundary

Table solvers still require categorical information sets. Supplied deterministic functions and fitted linear-softmax rules can use continuous observations during forward evaluation:

- [`DecisionNode`](../../pylimid/graph/decision_node.py) is configured by its action `states`. A configured discrete decision can have a continuous information parent; adding that parent does not turn it into a continuous decision or leave it unconfigured. `states=None` still means unconfigured.
- [`Policy`](../../pylimid/policy.py) accepts action tables, JAX-traceable deterministic functions, and fitted `SoftmaxPolicy` rules. Table compilation and `_info_assignments` still require named finite parent states and materialize their Cartesian product. Function rules receive raw parent values by keyword and return scalar integer action indices.
- [The scan](../../pylimid/solver/scan/batched.py) and [backward induction](../../pylimid/solver/backward_induction.py) consume those tables. Their continuous chance-node support does not extend to continuous information sets.
- [`Solution`](../../pylimid/solver/solution.py) accepts table and function rules and renders functions without enumerating their observations. Table solvers return tables; policy-gradient training returns a fitted stochastic rule.
- [`infer(policy=...)`](../../pylimid/inference/api.py) accepts fixed integer actions, tables, deterministic functions, and fitted stochastic rules. [`to_model`](../../pylimid/numpyro/model.py) evaluates contingent rules through one interface. Function or fitted stochastic rules combined with chance-node evidence raise `NotImplementedError`; existing table posterior inference is unchanged. Forward sampling checks function outputs against the action space.

[`DecisionNode.policy_family`](../../pylimid/graph/decision_node.py) accepts `LinearSoftmax()`. [`solve(method="policy_gradient")`](../../pylimid/solver/policy_gradient.py) fits exactly one such decision using sampled-action score gradients and Adam, then evaluates the final stochastic policy on independent scenarios. Numerical parents supply raw flattened features; categorical parents supply one-hot features. The fitted rule preserves feature order, parent shapes, categorical states and action labels. No scaling is fitted automatically.

`num_steps` and `learning_rate` configure training; `num_samples` is the batch size and final evaluation size. The previous batch's mean return supplies an independent baseline for the next update. The first baseline is zero. The solver returns the final iterate without candidate selection or convergence diagnostics. `method="auto"` still selects table solvers; those reject declared families rather than silently replacing them. Existing table behavior and guarantees remain the reference for categorical information sets.

## What can still be enumerated

The continuous observation space cannot be exhaustively tabulated. An unrestricted space of policies over those observations cannot be scanned either. The finite action space can still be enumerated at a given observation.

For one decision, let `Q(h, a)` be the expected total utility of choosing action `a` after observing `h`. If `Q` is available, an optimal action at that observation is simply `argmax_a Q(h, a)`. No gradient-based policy training is required to make that one comparison. Obtaining conditional values at arbitrary continuous observations can itself require inference, regression, or conditional simulation; grouping forward draws by exactly equal observations will not supply them.

Learning a reusable policy over the observation space is a different task. This proposal optimizes parameters within creator-declared policy families. Learning action-value functions and taking their argmax, or discretizing observations and using a table, remain alternative approaches for other solvers.

## The creator chooses the form; the solver fits it

When defining a decision, the creator declares its available actions, information parents, and policy family. The library should provide built-in families so choosing a representation does not require writing an optimizer. Custom families must meet the selected solver's contract. The solver fits parameters within the declared family; it does not choose a different architecture.

For example, an oil-drilling decision can declare `policy_family=LinearSoftmax()`. The neural-network alternative below illustrates a future family, not an implemented API:

| Part of the decision | Creator chooses | Solver determines |
| --- | --- | --- |
| Actions | `do_not_drill`, `drill` | Which action the fitted policy selects or samples at each observation |
| Information | The continuous seismic reading | How the policy uses that reading within the chosen family |
| Policy family | Linear scores followed by softmax | Weights and biases |
| Alternative policy family | A neural network followed by softmax, with a chosen architecture | Network weights and biases |

The probability of oil given a seismic reading comes from the chance model. The decision policy describes how to act on that reading. Choosing the form of the decision policy does not mean supplying its fitted parameters or the final drilling strategy.

Softmax converts scores into action probabilities. For `K` actions, a creator-selected linear family could use:

```text
logits(h; theta) = W features(h) + b
pi_theta(a | h) = softmax(logits(h; theta))[a]
```

A future extension could let the creator choose nonlinear features or a neural network to produce the logits. A more expressive family permits more complex strategies and can require more optimization and samples. The solver searches the selected family; it cannot recover a strategy that family cannot represent.

### One policy abstraction

A `Policy` contains one rule per decision. A table implements a rule by indexing categorical parent states; a function computes from observed parent values. Both use the same policy evaluation interface, and a diagram may mix the two. Preserve the existing table input format and its inspectable entries. Under the table family, the solver chooses an action for every information-set assignment.

The result contains each fitted rule together with its family, parameter values, action labels, and information-parent binding. Feature encoding and any fitted scaling must travel with the rule. Mixed information sets, such as temperature plus an earlier discrete decision, need explicit categorical encodings; treating a category index as a numerical feature would impose an ordering on its labels. A stochastic rule also needs an explicit random key when sampling actions.

A supplied function with fixed parameters is already a complete policy: it can be evaluated without solving. A trainable family additionally exposes parameters and the operations required by its optimizer. For example, a deterministic threshold can be evaluated using the existing JAX-traceable simulation contract, but optimizing its cutoff requires a compatible search method. A sampled threshold objective can have zero derivatives almost everywhere, so it is not automatically supported by the score-function solver below.

The creator chooses the family on `DecisionNode.policy_family`; `LinearSoftmax` is implemented and `SoftmaxPolicy` preserves the fitted rule. The solver fits parameters and does not choose another representation. Contracts for additional custom families and optimizer diagnostics remain open.

## A general approximate solver

The candidate solver applies the same procedure to different finite acyclic diagrams, including diagrams with several decisions, using the creator-declared policy families. It requires differentiable action log probabilities; an incompatible family needs a different solver. Let `theta` collect the policy parameters, and let `p_theta` be the joint distribution generated by the chance model and those policies:

```text
R = sum of all utility nodes in a simulated scenario
J(theta) = E_{p_theta}[R]
maximize J(theta) over theta
```

The chance distributions and utility callables are fixed during solving. Parameters enter through the decision policies. This assumption matters for the gradient formula below; additional direct model or utility parameters would need their own derivative terms.

### Sampled actions: score-function policy gradient

The broadly applicable baseline samples each action from its policy and uses a likelihood-ratio, or score-function, estimator. For each decision `i`, with local parameters `theta_i`, observed information `H_i`, and sampled action `A_i`:

```text
grad_{theta_i} J = E[R * grad_{theta_i} log pi_i(A_i | H_i; theta_i)]
```

This identity requires the usual integrability and differentiation-under-the-expectation conditions. Softmax policies have fixed finite action support at finite parameters. Finite expected utility is required to define the objective; finite gradient variance is an additional practical concern.

The derivative of the log probability holds the realized observation and action fixed. In an autodiff implementation, the sampled utility and observations are stopped from contributing pathwise derivatives to this score term. Applying `jax.grad` directly to average sampled utility does not produce this estimator: a categorical sample cannot be differentiated as though its integer output changed smoothly.

The estimator differentiates policy probabilities, not the environment. Chance distributions and utility functions must remain executable and JAX-traceable for the proposed batched walk, but they need not be differentiable with respect to a discrete action. Discrete chance descendants, action-indexed tables, and discontinuous sampled utilities can therefore be simulated without requiring a pathwise derivative through them.

Subtract an action-independent baseline to reduce variance. A constant fitted on separate data, or a learned function of `H_i` fitted on earlier or independent batches, is valid; an arbitrary quantity depending on the sampled action is not. Fitting a baseline to the same returns used in a score estimate can introduce dependence on those actions and bias, even if the baseline is called a constant. Baseline fitting and the score term must have separate derivative treatment. Variance reduction is part of making the solver usable, not evidence that it finds the global optimum.

For several decisions, walk the whole graph under the current joint policy and update their parameters. Earlier actions can change later observations, so a fixed dataset of independently drawn parent values is not generally a valid training distribution. The full-scenario return gives a simple estimator; using only utility descendants of a decision can reduce variance when justified by the graph.

### Enumerated actions: differentiable expected utility

For one decision whose observation distribution does not depend on its own parameters:

```text
J(theta) = E_H[sum_a pi_theta(a | H) * Q(H, a)]
```

Enumerate the finite actions and differentiate their probability weights. This avoids sampling the action and can reduce variance. `Q` can be estimated by conditional continuation simulation where available; it need not be a differentiable function of the integer action. A learned `Q` introduces approximation error in addition to Monte Carlo error.

With several decisions, exact branching over actions must also regenerate affected downstream chance variables and observations. Reusing a later observation from another action branch can change the decision problem. Branching can grow exponentially, and NumPyro's latent-variable enumeration does not automatically supply a utility objective or enforce these policy semantics.

The proposed first general optimizer uses sampled actions with score-function gradients. Conditional action enumeration is an optional variance-reduction improvement once its continuation semantics and cost are validated. Neither approach enumerates all continuous observations or all policies.

### Optimization and evaluation loop

1. Check that the declared policy families support the solver, then initialize their parameters and any baseline parameters.
2. Draw a fresh batch of complete scenarios under the current policies, respecting each decision's information set.
3. Estimate expected utility and score-function gradients, and update the policy parameters with an optimizer such as Adam. Fit a learned baseline separately if used.
4. Repeat within a stated optimization budget; use several initializations when useful. Estimated utility need not improve on every stochastic update.
5. Select candidate policies using separate validation simulations. Report final expected utility on fresh evaluation simulations that were not used for training or selection.

Common random numbers can help compare candidates, but repeatedly optimizing a fixed finite simulation bank can overfit it. Evaluation should describe Monte Carlo uncertainty when the required moments exist, initialization sensitivity, and the training budget. Reaching the iteration limit is not evidence of global optimality or even convergence.

## NumPyro and JAX responsibilities

NumPyro already permits distribution parameters to be functions of continuous values. It can sample chance nodes and categorical policy actions; JAX provides compilation, batching, and differentiation. A score-function surrogate can be differentiated by JAX, with an optimizer updating the policy parameters.

The missing layer is influence-diagram policy optimization: assemble local policies, respect information arcs, evaluate all utility nodes, form the correct gradient estimator, and return reusable policies and their evaluation. Existing inference traces and enumeration annotations are useful machinery, not a ready-made solver.

Ordinary SVI minimizes a negative ELBO, not negative expected utility. Adding `numpyro.factor("utility", R)` to a probabilistic model does not by itself establish an equivalent maximum-expected-utility solver; inference may introduce entropy or other objective terms. This proposal uses an explicit expected-utility objective and gradient estimator rather than treating utility as ordinary evidence.

## Worked oracle: a continuous signal and two actions

Use a fully observed scalar signal and a binary decision:

```text
X ~ Uniform(-1, 1)
A in {0, 1}, with X in its information set
U(X, A) = A * X
```

The optimal deterministic policy chooses action 1 when `X > 0`, action 0 when `X < 0`, and either action at zero. Its expected utility is:

```text
J* = (1/2) * integral from 0 to 1 of x dx = 1/4
```

Without observing `X`, every fixed action and every randomized policy independent of `X` has expected utility zero. This gives a direct value-of-information comparison.

For this oracle, the creator selects the binary linear-softmax family, equivalently `pi_theta(1 | x) = sigmoid(w*x + b)`. The solver fits `w` and `b`. The enumerated objective is `E[X * sigmoid(w*X + b)]`. At `w = b = 0`, its exact gradient is `(dJ/dw, dJ/db) = (1/12, 0)`. This checks the score-function estimator against an analytic derivative, not merely another implementation of the same sampled loss.

With `b = 0`, increasing positive `w` approaches the optimal threshold. The stochastic sigmoid policy reaches `1/4` only as `w` tends to infinity; validation must compare utility within tolerance rather than require a finite optimizer coefficient matching the deterministic rule.

Taking the largest logit produces a deterministic policy, but changes the policy being evaluated. In a general multi-decision problem, independently replacing every randomized rule with its argmax need not improve utility. A result must preserve the trained policy semantics; any deterministic conversion needs separate evaluation.

## Validation plan

The single-decision tests in `tests/solver/test_policy_gradient.py` cover the analytic and action-enumerated gradient oracle, learned utility, mixed categorical/vector observations, hidden non-parent information, discrete descendants, stochastic action/descendant correlation, reproducibility, policy reuse and unsupported solver selections. The broader validation plan below also includes future stages:

- **Analytic oracle.** Recover the signal example's utility within stated tolerances and sample budgets. At known parameters, compare Monte Carlo score gradients to `(1/12, 0)` and to action-enumerated gradients.
- **Categorical overlap.** On small existing categorical fixtures, compare against the exhaustive scan, which supplies a finite-policy reference. Include a table-logit policy family capable of approximating each deterministic table; an affine classifier alone may not represent that reference. An approximate optimizer is not required to reach the reference on every run.
- **Mixed parents and information constraints.** Combine a continuous signal with a categorical parent, and include a useful hidden ancestor that is deliberately not observed. Check that policy evaluation only receives declared parents.
- **Multiple decisions.** Include an earlier action affecting a later observation. Compare sampled-action gradients with exact finite action branching on a small analytically tractable model. Include a LIMID without full memory so the solver cannot silently use a full-history policy or assume a one-pass backward solution.
- **Discrete descendants and utility discontinuities.** Exercise a sampled categorical consequence and an integer-indexed or indicator payoff. Verify the score estimator against an analytic or enumerated expectation despite a zero or undefined pathwise derivative.
- **Policy families and reuse.** Evaluate returned table and function rules through the same interface, including in one diagram. Preserve the creator-selected family, fitted parameters, action labels, information order, feature encoding, and stochastic or deterministic meaning. Reject incompatible solver/family combinations and policy bindings explicitly; do not silently substitute a different family.
- **Training and evaluation separation.** Check reproducibility for a fixed key schedule, initialization sensitivity, and held-out utility estimates. Do not assert monotone improvement on every optimizer step or claim coverage of every continuous observation from finite draws.

## Guarantees and limitations

The proposed solver is general across supported graph structures, not universal across arbitrary models or policy functions. It requires a sampleable finite acyclic model, finite discrete actions, policies with evaluable differentiable log probabilities, and sufficient regularity for its gradient estimator. Its practical reliability also depends on variance, observation dimension, rare events, feature choice, and optimization budget.

There are separate sources of error: policy-family approximation, imperfect optimization, and Monte Carlo estimation. More simulations address sampling error but do not remove the first two. A held-out utility interval describes evaluation uncertainty, not the gap to the globally optimal unrestricted policy. Report an approximate candidate, without promising even a local optimum merely because training stopped.

The existing scan's exhaustive finite-policy search guarantee does not carry over. Nor does structural solubility alone make a learned function policy globally optimal: any backward-induction extension with regression or restricted policies needs its own approximation analysis.

## Proposed implementation stages and open decisions

1. **Evaluate supplied function policies (implemented).** Keep finite action `states`; use one policy abstraction with table and deterministic function rules evaluated only from declared information parents. Support forward evaluation and rendering, preserving the existing table input format. No optimizer is needed to evaluate a supplied rule. Randomized rules and posterior inference through functions remain outside this step.
2. **Optimize one discrete decision (implemented).** Declare `DecisionNode.policy_family=LinearSoftmax()` and select `method="policy_gradient"`. Fit score gradients with Adam, a previous-batch baseline, and explicit budgets; evaluate the final stochastic policy on fresh simulations. Analytic and action-enumerated gradients validate the estimator. The returned `SoftmaxPolicy` supports stochastic forward reuse. Multi-decision training and other trainable families are rejected.
3. **Optimize several decisions.** Extend the validated estimator to full trajectories with mixed information sets and explicit LIMID memory. Add baselines and report candidate selection and evaluation separately. This is the general solver milestone; one-decision support alone is not sufficient.
4. **Improve efficiency.** Investigate conditional action enumeration, richer policy families, or learned action-value policies where benchmarks justify them.
5. **Revisit continuous actions independently.** Reuse policy binding and evaluation infrastructure where appropriate, while adding action domains and suitable pathwise or other estimators as separate work.

Before extending training, decide the custom-family contract and which optimizer diagnostics belong in the result. Fixed actions, tables, deterministic functions and fitted stochastic rules already have separate dispatch. The creator's family is part of the decision specification. Extending `method="auto"` must preserve that choice and requires an agreed approximation and result contract.

Forward evaluation should be supported before promising posterior reuse through `infer(policy=...)`. Conditioning with a threshold policy can introduce discontinuities in continuous latent variables, so accepting a callable does not automatically make the current NUTS path reliable. Posterior inference compatibility needs its own documented support boundary.

Out of scope: continuous actions, action bounds, data fitting, posterior-to-planning transfer, learning an unknown environment, cyclic or infinite-horizon models, and global-optimality certificates. The model is given; simulation-based policy optimization does not require an online reinforcement-learning interface.

## References

- [Li and Shenoy (2010), *Solving Hybrid Influence Diagrams with Deterministic Variables*](https://proceedings.mlr.press/r8/li10a.html): established hybrid influence-diagram semantics and solution methods. Supports the modeling case; it is not a claim that the proposed optimizer implements their algorithm.
- [Lauritzen and Nilsson (2001), *Representing and Solving Decision Problems with Limited Information*](https://pubsonline.informs.org/doi/10.1287/mnsc.47.9.1235.9779): local information sets, single policy updating, and the conditions behind exact solution orderings.
- [Schulman et al. (2015), *Gradient Estimation Using Stochastic Computation Graphs*](https://arxiv.org/abs/1506.05254): score-function and pathwise gradient estimators, surrogate objectives, and variance reduction in stochastic directed acyclic graphs.
- [NumPyro primitives](https://num.pyro.ai/en/stable/primitives.html) and [SVI objectives](https://num.pyro.ai/en/stable/svi.html): the available probabilistic machinery and the distinction between an inference objective and expected-utility optimization.
