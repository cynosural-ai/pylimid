# Posterior inference with fitted categorical policies

Status: conditioning and acceptance checks are implemented through the existing `infer` API, including policy batching and action enumeration. This note records its behavior contract and validation results. Published documentation updates and the inference guide revision follow separately.

## Goal

Allow `infer(diagram, query=..., observed=..., policy=solution.policy)` to condition on chance-node evidence when relevant decisions use fitted `CategoricalPolicy` rules. Information parents may be categorical, continuous, or mixed, including numerical arrays with their fitted shapes. Numerical count information remains subject to the existing posterior count restrictions.

The extension applies to both built-in families, `TabularSoftmax` and `LinearSoftmax`, and compatible custom families through the shared fitted-policy interface. It supports parentless decisions and multiple decisions, including explicit memory of earlier actions and observations.

## Conditioning semantics

A fitted policy contributes a conditional action distribution to the joint model, just as a chance node contributes a conditional distribution. For chance variables `X`, actions `A`, and the fitted joint policy `pi`, the joint density or mass is:

```text
p_pi(X, A) = product over chance nodes of p(x | parents)
             * product over fitted decisions of pi(a | information parents)
```

Fixed actions and deterministic action tables retain their existing semantics. Conditioning on chance evidence restricts this same joint model and normalizes over the remaining unknowns. In particular, evidence downstream of a decision can change the posterior distribution of its action and information parents. Fitted parameters remain fixed during inference, and each policy uses only its declared information parents.

For a model with `temperature -> action -> outcome` and `temperature -> outcome`:

| Evidence | Required behavior |
| --- | --- |
| No evidence | Sample temperature, then action from the fitted policy, then outcome. |
| Temperature observed | Evaluate action probabilities at that temperature and infer outcome. |
| Outcome observed | Infer temperature while summing over actions with their fitted probabilities and outcome likelihoods. |
| Temperature and outcome observed | Weight actions using both the fitted probabilities at that temperature and the observed outcome likelihood. |

These cases also apply to categorical and mixed information. Observing a decision's parent through `observed=` is distinct from declaring that parent in the decision's information set.

## Public API and results

The extension uses the existing `infer` signature and fitted policy representation. Relevant fitted rules may be combined with chance-node evidence, removing their current blanket rejection. The existing ancestor pruning and binding requirements continue to apply: every relevant decision needs a policy, and irrelevant decisions do not participate in inference.

Queries continue to name chance and utility nodes. Evidence continues to name chance nodes; fixed actions are supplied through `policy=`. Decisions become latent sample sites internally when their fitted policies are used, but exposing decision posteriors through `query=` is a separate API change. This contract does not expand the public evidence-value types; array-valued information must be handled when present as latent or otherwise supported parent values.

Returned `Posterior` arrays retain their sample axis, node shapes, categorical labels, and aligned joint draw order. Queried utilities are evaluated from those joint draws. Returned summaries remain Monte Carlo estimates, including when enumeration supplies the draws.

## Policy evaluation and shapes

Policy families continue to compute one action-logit vector for one information assignment. The public `CategoricalPolicy` observation methods retain that same contract. Existing family implementations and custom families should not need to understand NumPyro enumeration axes.

An internal inference adapter handles leading enumeration and batch dimensions. It separates those dimensions from each parent's fitted value shape, broadcasts the leading dimensions across parents, evaluates the policy for each assignment, and preserves a final action axis in the resulting categorical distribution. Parentless policies produce a single action distribution.

For example, a fitted numerical parent with shape `(3,)` still supplies a three-component vector to the family even when several categorical assignments are enumerated. Enumeration axes must not become additional numerical features. Parents enumerated on different axes must retain their joint combinations, rather than being paired element by element.

The adapter preserves declared parent order and the existing checks for parent names, labels, action labels, and fitted shapes. Invalid value shapes remain errors. Gradients through continuous information must reach the policy probabilities used in the marginalized likelihood.

## Posterior execution

Fitted categorical decisions participate in discrete enumeration alongside supported discrete chance nodes. With no continuous latents, the existing enumeration path supplies posterior draws. With continuous latents, NUTS samples the continuous posterior with discrete unknowns marginalized out; the conditional sampling pass recovers discrete chance and action draws given each continuous draw and the evidence.

Multiple decisions use the same joint model. A later policy can depend on an earlier action only through its declared parents. Enumeration must account for every relevant action distribution, including when earlier actions affect later continuous observations.

Smooth fitted policies such as `LinearSoftmax` fit the existing gradient-based posterior approach when the rest of the model is compatible with NUTS. A custom family must support JAX transformations and, where continuous latent information enters the likelihood, differentiation with respect to that information. Compatibility with training gradients alone does not guarantee compatibility with NUTS, and the extension supplies no general convergence guarantee.

## Boundaries

- Arbitrary deterministic function policies retain their current evidence restriction; conditioning through discontinuous functions is separate work.
- Existing fixed-action and deterministic-table conditioning remains supported.
- Count latents retain the current scalar, enumerable, fixed zero-based support requirements. Unbounded, shifted, batched, and non-static count supports gain no new posterior support.
- Continuous actions, new posterior algorithms, and broader public query or evidence types are outside this extension.
- Chance distributions must remain compatible with NumPyro enumeration and JAX transformations. Supporting fitted policies does not make arbitrary downstream distribution callables enumeration-compatible.

## Acceptance checks

Validate the following behaviors with exact sums, numerical integration on small models, direct checks, and the existing test suite.

- A stochastic categorical table conditioned on an outcome agrees with an explicit sum over chance states and actions.
- Observing a continuous information parent uses the fitted probabilities at the supplied value.
- Observing a downstream outcome correctly updates an unknown continuous information parent and action-dependent utility; nonuniform policies and action-dependent likelihoods expose missing policy weights.
- Mixed categorical and numerical information, vector-valued numerical information, and multiple independently enumerated parents preserve shapes and posterior weights.
- Multiple decisions, remembered actions, and action-dependent continuous observations preserve the joint dependencies.
- Parentless fitted rules, observed parents, and irrelevant policies follow the existing pruning and binding behavior.
- Forward evaluation, fixed actions, deterministic tables, utilities, sample alignment, and existing unsupported-input behavior retain their established contracts.

## Validation results

Acceptance checks completed on 2026-10-27. The [fitted-policy regression module](../../tests/inference/test_fitted_policies.py) adds six cases covering:

- The full discrete joint posterior for two fitted decisions, mixed policy families, differently sized categorical parents, declared parent order, remembered actions, and aligned utility draws, compared with an explicit probability sum.
- Mixed categorical and vector information under `LinearSoftmax`, and matrix information under a nonlinear custom family with nested parameters, compared with two-dimensional Gaussian quadrature. The categorical information parent itself depends on the continuous signal. Checks include continuous and categorical posterior summaries, action-dependent utility, retained array shapes, sample alignment, and unchanged fitted parameters.
- An observed continuous information parent, with and without an observed downstream outcome, compared with the conditional action probabilities.
- Multiple fitted decisions with an earlier action affecting later continuous information, compared with Gaussian quadrature for marginal and joint moments.

Additional direct checks confirmed finite fixed-bound Binomial information and joint utilities, pruning of irrelevant fitted policies, zero-probability actions, continued rejection of unsupported Poisson and batched count latents, and rejection of changed numerical information shapes. Earlier public-API checks also reused a policy returned by `solve` and compared its posterior results with Gaussian quadrature.

The full suite passed with 410 tests, including the existing function-policy, fixed-action, deterministic-table, count-domain, and API-boundary checks. Ruff lint, Ruff format checking, and ty checking passed. These checks found no additional implementation gaps within the declared scope; the existing backend boundaries above still apply.

Published documentation and inference-page examples are updated afterwards.
