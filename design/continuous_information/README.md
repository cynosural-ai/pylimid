# Discrete decisions with continuous information

**Status:** proposal, not implemented. This is a separate extension from continuous actions and can be implemented first. Every decision in scope still declares a finite tuple of action `states`; what changes is the policy's ability to use continuous observations.

## Scope and sequencing

An action space and an information set are different parts of a decision problem:

| Actions | Observations | Policy representation | Status |
| --- | --- | --- | --- |
| Discrete | Finite categorical states | Table from parent-state assignments to action indices | Implemented |
| Discrete | Continuous, or mixed categorical and continuous | Function from observed values to action indices or action probabilities | This proposal |
| Continuous | Categorical, continuous, or mixed | Function returning a real-valued action | Separate continuous-action proposal |

Observing an exact temperature and choosing **fan on/off** belongs to this proposal. Observing temperature and choosing **heater power** requires continuous actions. Continuous chance variables elsewhere in the graph are already supported and do not by themselves require either extension.

The existing [continuous-decision draft](https://github.com/cynosural-ai/pylimid/pull/41) groups both missing capabilities into a wider exploration. This note separates the discrete-action case so its policy representation, optimizer, and validation can be reviewed and implemented without declaring continuous decisions, action bounds, or differentiable action-to-utility paths.

Suggested order: support evaluating policies that accept continuous information, add an approximate solver for those policies, then revisit continuous actions. This is a proposed order, not a claim that the two action types use the same gradient estimator.

## Influence-diagram semantics

An arc into a decision means that the parent value is observed before acting. A continuous observation does not change that meaning: the decision rule can be a threshold, a piecewise function, or another function of the observed value. Hybrid influence diagrams with continuous chance variables and discrete decisions are established in the literature; see the [references](#references).

A deterministic policy is an action-valued function `a = delta(h)`, where `h` contains exactly the decision's information parents. A randomized policy is a conditional distribution `pi(a | h)`. A softmax is one possible parameterization of that distribution, not a requirement imposed by the graph.

The policy parameters are chosen to maximize utility. They are not unknown chance-node parameters inferred from data. A decision keeps its controlled-choice semantics even if a solver samples its action during optimization.

Information constraints remain explicit. A policy may not inspect an unobserved ancestor, another node outside its information set, or remembered history unless the corresponding parents are drawn. Earlier decisions and observations are remembered only through those explicit arcs.

## Current implementation boundary

The restriction is in the table representation, not a mathematical prohibition on continuous information or a NumPyro distribution limitation:

- [`DecisionNode`](../../pylimid/graph/decision_node.py) is configured by its action `states`. A configured discrete decision can have a continuous information parent; adding that parent does not turn it into a continuous decision or leave it unconfigured. `states=None` still means unconfigured.
- [`Policy` and `_info_assignments`](../../pylimid/policy.py) require named finite parent states and materialize their Cartesian product. A continuous parent has no such table index. Numerical count parents without named states also fail this check, but their public support is a separate scope decision.
- [The scan](../../pylimid/solver/scan/batched.py) and [backward induction](../../pylimid/solver/backward_induction.py) consume those tables. Their continuous chance-node support does not extend to continuous information sets.
- [`Solution`](../../pylimid/solver/solution.py) stores integer action tables and renders every row. A function over continuous observations cannot be returned or printed using that contract alone.
- [`infer(policy=...)`](../../pylimid/inference/api.py) accepts fixed integer actions and action tables. [`to_model`](../../pylimid/numpyro/model.py) resolves contingent decisions by indexing those tables, and skips utility nodes. A policy optimizer needs utility evaluation as well as a more general policy binding.

This proposal does not implement any of these changes. Existing behavior and solver guarantees remain the reference for categorical information sets.

## What can still be enumerated

The continuous observation space cannot be exhaustively tabulated. An unrestricted space of policies over those observations cannot be scanned either. The finite action space can still be enumerated at a given observation.

For one decision, let `Q(h, a)` be the expected total utility of choosing action `a` after observing `h`. If `Q` is available, an optimal action at that observation is simply `argmax_a Q(h, a)`. No gradient-based policy training is required to make that one comparison. Obtaining conditional values at arbitrary continuous observations can itself require inference, regression, or conditional simulation; grouping forward draws by exactly equal observations will not supply them.

Learning a reusable policy over the observation space is a different task. A practical solver can optimize a finite parameter vector, learn approximations to the action-value functions and take their argmax, or discretize observations and use a table. Parameter optimization is the candidate developed here; the other approaches remain valid alternatives.

## Candidate policy representation

For a decision with `K` actions, a trainable policy could return `K` logits from its observed information:

```text
logits(h; theta) = W features(h) + b
pi_theta(a | h) = softmax(logits(h; theta))[a]
```

An affine feature model is a small starting family. Categorical parents need an explicit encoding; continuous parents may need scaling for stable optimization. The encoding and any fitted scaling belong to the returned policy so later evaluation uses the same transformation. Nonlinear features or a small neural network can increase expressiveness, with additional optimization and sample costs.

Mixed information sets should be supported deliberately: for example, temperature plus an earlier discrete decision. Using a scalar state index as a numerical feature would impose an ordering on category labels; use an explicit categorical encoding instead.

Supporting a supplied deterministic function, such as a threshold returning an integer action, is useful before adding optimization. Evaluating that function only requires the existing JAX-traceable simulation contract. Training an arbitrary threshold with ordinary autodiff is not implied: a sampled threshold objective can have zero derivatives almost everywhere.

There is no final public API in this note. The needed contract is a policy bound to a decision's declared parents and actions, with enough information to evaluate it again. A trained stochastic policy also needs an explicit random key when sampling actions.

## A general approximate solver

The candidate solver applies the same procedure to different finite acyclic diagrams, including diagrams with several decisions. Let `theta` collect the policy parameters, and let `p_theta` be the joint distribution generated by the chance model and those policies:

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

1. Initialize the policies and any baseline parameters.
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

For `pi_theta(1 | x) = sigmoid(w*x + b)`, the enumerated objective is `E[X * sigmoid(w*X + b)]`. At `w = b = 0`, its exact gradient is `(dJ/dw, dJ/db) = (1/12, 0)`. This checks the score-function estimator against an analytic derivative, not merely another implementation of the same sampled loss.

With `b = 0`, increasing positive `w` approaches the optimal threshold. The stochastic sigmoid policy reaches `1/4` only as `w` tends to infinity; validation must compare utility within tolerance rather than require a finite optimizer coefficient matching the deterministic rule.

Taking the largest logit produces a deterministic policy, but changes the policy being evaluated. In a general multi-decision problem, independently replacing every randomized rule with its argmax need not improve utility. A result must preserve the trained policy semantics; any deterministic conversion needs separate evaluation.

## Validation plan

These are requirements for a future implementation, not tests added by this documentation proposal:

- **Analytic oracle.** Recover the signal example's utility within stated tolerances and sample budgets. At known parameters, compare Monte Carlo score gradients to `(1/12, 0)` and to action-enumerated gradients.
- **Categorical overlap.** On small existing categorical fixtures, compare against the exhaustive scan, which supplies a finite-policy reference. Include a table-logit policy family capable of approximating each deterministic table; an affine classifier alone may not represent that reference. An approximate optimizer is not required to reach the reference on every run.
- **Mixed parents and information constraints.** Combine a continuous signal with a categorical parent, and include a useful hidden ancestor that is deliberately not observed. Check that policy evaluation only receives declared parents.
- **Multiple decisions.** Include an earlier action affecting a later observation. Compare sampled-action gradients with exact finite action branching on a small analytically tractable model. Include a LIMID without full memory so the solver cannot silently use a full-history policy or assume a one-pass backward solution.
- **Discrete descendants and utility discontinuities.** Exercise a sampled categorical consequence and an integer-indexed or indicator payoff. Verify the score estimator against an analytic or enumerated expectation despite a zero or undefined pathwise derivative.
- **Policy reuse.** Evaluate the returned policy with fresh keys and preserve its action labels, information order, feature encoding, and stochastic or deterministic meaning. Reject incompatible policy bindings explicitly.
- **Training and evaluation separation.** Check reproducibility for a fixed key schedule, initialization sensitivity, and held-out utility estimates. Do not assert monotone improvement on every optimizer step or claim coverage of every continuous observation from finite draws.

## Guarantees and limitations

The proposed solver is general across supported graph structures, not universal across arbitrary models or policy functions. It requires a sampleable finite acyclic model, finite discrete actions, policies with evaluable differentiable log probabilities, and sufficient regularity for its gradient estimator. Its practical reliability also depends on variance, observation dimension, rare events, feature choice, and optimization budget.

There are separate sources of error: policy-family approximation, imperfect optimization, and Monte Carlo estimation. More simulations address sampling error but do not remove the first two. A held-out utility interval describes evaluation uncertainty, not the gap to the globally optimal unrestricted policy. Report an approximate candidate, without promising even a local optimum merely because training stopped.

The existing scan's exhaustive finite-policy search guarantee does not carry over. Nor does structural solubility alone make a learned function policy globally optimal: any backward-induction extension with regression or restricted policies needs its own approximation analysis.

## Proposed implementation stages and open decisions

1. **Evaluate supplied function policies.** Keep finite action `states`; add a policy representation evaluated only from declared information parents, and define forward evaluation and rendering without pretending the observation space is a finite table. Preserve existing table policies. No optimizer is needed to evaluate a supplied rule.
2. **Optimize one discrete decision.** Establish the expected-utility objective and analytic oracle, then implement score gradients and compare with finite action enumeration. Choose a small policy family and explicit optimization settings.
3. **Optimize several decisions.** Extend the validated estimator to full trajectories with mixed information sets and explicit LIMID memory. Add baselines and report candidate selection and evaluation separately. This is the general solver milestone; one-decision support alone is not sufficient.
4. **Improve efficiency.** Investigate conditional action enumeration, richer policy families, or learned action-value policies where benchmarks justify them.
5. **Revisit continuous actions independently.** Reuse policy binding and evaluation infrastructure where appropriate, while adding action domains and suitable pathwise or other estimators as separate work.

Before implementation, decide the public policy/result representation, solver-owned versus user-supplied policy families, action sampling and random-key contracts, and which optimizer diagnostics belong in the result. Fixed actions, tables, and function policies need unambiguous dispatch. A new solver should initially be explicit; extending `method="auto"` requires an agreed approximation and result contract.

Forward evaluation should be supported before promising posterior reuse through `infer(policy=...)`. Conditioning with a threshold policy can introduce discontinuities in continuous latent variables, so accepting a callable does not automatically make the current NUTS path reliable. Posterior inference compatibility needs its own documented support boundary.

Out of scope: continuous actions, action bounds, data fitting, posterior-to-planning transfer, learning an unknown environment, cyclic or infinite-horizon models, and global-optimality certificates. The model is given; simulation-based policy optimization does not require an online reinforcement-learning interface.

## References

- [Li and Shenoy (2010), *Solving Hybrid Influence Diagrams with Deterministic Variables*](https://proceedings.mlr.press/r8/li10a.html): established hybrid influence-diagram semantics and solution methods. Supports the modeling case; it is not a claim that the proposed optimizer implements their algorithm.
- [Lauritzen and Nilsson (2001), *Representing and Solving Decision Problems with Limited Information*](https://pubsonline.informs.org/doi/10.1287/mnsc.47.9.1235.9779): local information sets, single policy updating, and the conditions behind exact solution orderings.
- [Schulman et al. (2015), *Gradient Estimation Using Stochastic Computation Graphs*](https://arxiv.org/abs/1506.05254): score-function and pathwise gradient estimators, surrogate objectives, and variance reduction in stochastic directed acyclic graphs.
- [NumPyro primitives](https://num.pyro.ai/en/stable/primitives.html) and [SVI objectives](https://num.pyro.ai/en/stable/svi.html): the available probabilistic machinery and the distinction between an inference objective and expected-utility optimization.
