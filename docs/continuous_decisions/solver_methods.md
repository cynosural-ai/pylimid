# Solving continuous decisions — methods and a worked example

Companion to [`README.md`](./README.md). This note makes the feasibility claims concrete: what is optimized, how continuous policies are parameterized, how gradients of an expectation are estimated, what the optimization loop looks like, and where the naive approach breaks (with fixes). It is a design map for Strategy A, not an implementation.

## What is optimized

Solving an influence diagram means choosing a policy `δ` for every decision and maximizing

```
J = E[U]
```

where the expectation is over the chance variables and the decisions are resolved by the policies. For a discrete decision the policy is a table, and the current solvers enumerate tables. For a continuous decision the policy is a function of the information set, and the space of functions is infinite-dimensional. The practical reduction: parameterize each policy with a finite parameter vector `θ` and maximize `J(θ) = E[U | θ]` instead.

Two existing library facts make this work:

- `UtilityNode.values` is already a deterministic callable, so a sampled scenario has a well-defined utility.
- The forward walks are ordinary JAX computation, so the entire simulation — sampling, policy evaluation, utility — is differentiable when the user callables are written in `jnp`.

The expectation is an integral with no closed form in general. Monte Carlo plus autodiff replaces it: the average utility over sampled scenarios estimates `J(θ)`, and `jax.grad` of that average estimates its gradient.

## Terminology and naming

Strategy A is the design-doc name for policy-as-parameters. For the all-continuous case the method is **deterministic policy optimization through a differentiable simulation**: a deterministic policy is optimized by pathwise gradients through a known probabilistic model. This note, and the eventual solver, uses that name rather than "policy gradient" deliberately:

- In the literature, **policy gradient** conventionally means the score-function family (REINFORCE and its descendants) for stochastic policies and unknown dynamics — the high-variance estimator discussed below.
- The all-continuous case uses the **pathwise/deterministic** gradient, which is closer to differentiable simulation than to REINFORCE. Calling it policy gradient would misdescribe the algorithm and set the wrong expectations about variance and sample counts.
- The score-function estimator becomes relevant only when a discrete selection sits on the differentiated path — discrete decisions with continuous information, or discrete chance descendants. When it arrives, that specific estimator is best named **score-function policy gradient**, while the solver and method stay **policy optimization**.

Intended naming: the solver module is `solvers/policy_optimization.py`; the discrete table machinery in `solvers/_policy.py` is unchanged, and the public design name remains Strategy A.

## Policy parameterization

- **Affine** — `δ(s) = a + b·s` (or `a + Σ b_i s_i` for several information parents). The natural first family: exact for LQG, one parameter vector per decision, cheap and stable. The worked example below uses it.
- **Bounded actions** — for an action in `[lo, hi]`, parameterize unconstrained and squash, e.g. `lo + (hi - lo)·sigmoid(φ)`. Unbounded actions use the raw parameter.
- **Discrete action with continuous information** — a classifier `π(action | info; θ)`, e.g. softmax over logits that are affine in the information. The policy returns a distribution, and solving needs the score function (below), not pathwise gradients.
- **Richer families** — quadratic, small MLP, or a user-supplied factory. More expressive, harder to train, more samples, more local optima; justified only when affine is demonstrably too weak.

A discrete-action policy with a *discrete* information set remains the current table case and is still solved by enumeration, not by this machinery.

## Gradient estimators

This is the heart of Strategy A. There are three tools, and they compose.

### Pathwise (reparameterization)

Write each chance draw as a differentiable function of fixed noise. A Normal draw is `μ + σ·z` with `z ~ N(0,1)`; then the scenario, and hence the utility, is a smooth function of `θ`, and

```
∇J(θ) = E[ ∇θ U(scenario(θ, z)) ]
```

an average of ordinary derivatives — no probability scores. This is the low-variance estimator and the preferred tool whenever the path from parameters to utility is differentiable. NumPyro distributions sample this way for the location-scale families, and implicit reparameterization (Figurnov, Mohamed & Mnih 2018) extends it to Gamma, Beta, Dirichlet, and others. A non-reparameterizable continuous distribution needs the implicit method where available, or falls back to the score function.

### Score function (REINFORCE)

When a branch is non-differentiable — a discrete action is selected, or a discrete chance node is sampled — differentiating through the value is impossible, but differentiating through its *probability* works:

```
∇J(θ) = E[ U · ∇θ log p(branch | θ) ]
```

Unbiased but high variance; a baseline (subtract a constant or a learned value estimate) is usually needed to make it practical. In this library it is the fallback for discrete decisions with continuous information, and the alternative when enumerating a discrete chance node is too expensive.

### Exact enumeration of discrete descendants

For small discrete nodes the better answer is not an estimator at all: sum them out inside the objective. For `d -> s -> U` with `s` discrete,

```
E[U | d] = Σ_s P(s | d) · U(d, s)
```

which is a smooth function of `d` whenever `P(s | d)` is smooth. This is the same enumeration the discrete solver performs over whole policies, applied locally to a chance node; its cost is the product of the enumerated state counts. NumPyro's parallel enumeration machinery does this for a model trace, and a hand-rolled differentiable walk can branch at discrete sites and weight the branches.

### Combining

The estimators compose along the causal path: pathwise through every differentiable segment, score/enumeration at every discrete or non-reparameterizable branch point. The rule from [`README.md`](./README.md) is the guide — discreteness *off* the differentiated path needs nothing at all.

## The optimization loop

1. Initialize policy parameters (small random values).
2. Draw a batch of scenarios under a fixed key schedule. Common random numbers across steps keep the objective stable and the gradients comparable.
3. Resolve each decision from its information set under the current policy; walk the graph to the utilities.
4. Average the utilities → `J` estimate. Take `jax.grad`; update the parameters with Adam.
5. Repeat for a fixed number of steps; optionally restart from different initializations and keep the best final `J`.
6. Report a final, larger-sample estimate of `J` for the chosen policy.

Two notes:

- **No posterior inference is needed during solving.** The solver simulates the joint distribution and evaluates the policy on realized information; `infer()` remains the path for conditioning on evidence.
- **Differentiability is stronger than traceability.** The batched scan already requires `values` and `dist` callables to be JAX-traceable; gradient solving additionally requires them to be differentiable — `jnp` operations, no `float()`/`int()` coercion, no data-dependent Python control flow.

## NumPyro's role and what is missing

NumPyro supplies the pieces: distributions with reparameterized and implicit-reparameterized sampling, JAX autodiff, parallel enumeration machinery, the existing vmapped/jitted walks, and `numpyro.optim` (optax is not currently a dependency but is the standard, small alternative). What it does not supply is the objective: SVI optimizes an ELBO, a bound on log evidence, not expected utility. The solver is therefore a custom training loop over `jax.grad` of a sampled `E[U]`, not `numpyro.infer.SVI`.

## Worked example: the smart thermostat

An all-continuous influence diagram with one continuous decision and a continuous information set.

```
X ~ Normal(loc=0, scale=4)        # how cold it is outside
S ~ Normal(loc=X, scale=2)        # noisy forecast, observed before deciding
a = δ(S)                          # decision: heater power
T ~ Normal(loc=X + a, scale=1)    # indoor temperature
U = -(T - 20)^2 - a^2             # comfort penalty + energy cost
```

Policy and objective:

```
δ(s) = α + β·s
J(α, β) = E[U]
```

### The closed-form answer (LQG)

Given the forecast, the posterior mean of `X` is `E[X | S] = 0.8·S` (the reliability of the forecast). Substituting and minimizing gives the optimal rule

```
a = 10 - 0.4·S
```

so `α = 10`, `β = -0.4`, with `J ≈ -210.6`. The response coefficient is `-0.4 = -0.5 × 0.8`: perfect information would give `-0.5`, and `0.8` is how informative the forecast is. The policy leans on the forecast exactly as much as it deserves. This is the certainty-equivalence property of LQG: the optimal action is the deterministic optimum evaluated at the conditional mean, which is linear in the observation — hence an affine policy is sufficient.

### What the gradient loop would do

Sample `X`, the forecast noise, and the furnace noise; compute `S`, `a = α + β·S`, `T`, and `U`; average `U`; differentiate with respect to `α` and `β`; take an Adam step. The reparameterized draws make `T` and `U` smooth in `α` and `β`, so the gradient is informative. With enough samples and steps the parameters converge to the analytic values above — which is exactly the validation test for the solver.

### Value of information

| Information | Optimal rule | `E[U]` |
| --- | --- | --- |
| nothing | `a = 10` | −217.0 |
| noisy forecast `S` | `a = 10 - 0.4·S` | −210.6 |
| true `X` | `a = 10 - 0.5·X` | −209.0 |

The ordering — no information ≤ noisy information ≤ full information — is a useful Monte-Carlo invariant.

## Utility pathologies and fixes

The mathematical fixes are here; the callable contract they depend on — traceability vs differentiability, and how the library should enforce it — is in [`utility_contract.md`](./utility_contract.md).

### Bounds

A real action usually has a domain: invest `[0, 100]`, dose `[0, mg_max]`. Parameterize unconstrained and squash (sigmoid times the range). Without a support, optimization can wander into physically meaningless values.

### Kinks

The newsvendor objective `U = p·min(D, a) - c·a` has a kink at `D = a`. The per-sample derivative is a subgradient, not a classical one. The *expectation* is nonetheless smooth — the kink's location moves with the random demand — and the optimum is known analytically: order the critical fractile of demand, `P(D <= a*) = (p - c)/p`. With `p = 1`, `c = 0.3`, the optimizer should find the 70th percentile of `D`. This is a non-LQG oracle (kinked, non-quadratic, non-Gaussian-friendly) that complements the thermostat.

### Discontinuities

An indicator utility such as `1{a >= D}` ("bonus if demand is covered") makes the naive pathwise derivative zero almost everywhere, even though the expectation is a perfectly smooth CDF. The estimator, not the objective, is the problem. Fixes, in order of preference: write the expectation analytically (`P(D <= a)`), enumerate the discrete event if it comes from a discrete node, or smooth the transition (for example `sigmoid((a - D)/τ)` with an annealed `τ`), or use the score function.

### Non-convexity

Beyond LQG, `J(θ)` may have local optima. Mitigations: several random restarts, keep the best final estimate; start from a sensible initialization; keep the policy family simple. There is no global guarantee, and none should be promised.

### High-dimensional information

A `k`-dimensional continuous information set cannot be tabulated; the policy must generalize. Affine policies over informative features are the stable default; richer models need more samples and risk overfitting the simulation. This is the continuous-decision face of the curse of dimensionality.

## Mixed models at a glance

The topology rules from [`README.md`](./README.md), applied to this loop:

- **Discrete decision, discrete information** — enumerate; everything downstream stays pathwise. This is the current solver, and it can be reused as the outer loop for mixed diagrams.
- **Discrete decision, continuous information** — classifier policy plus score function, or discretize the observation and enumerate.
- **Discrete chance descendant of a continuous decision** — enumerate the node inside the objective (preferred for small state spaces) or use the score function.
- **Non-reparameterizable continuous chance** — implicit reparameterization where NumPyro provides it, score function otherwise.
- **Mixed decisions** — compose: enumerate the discrete branches, optimize continuous policies inside (or jointly optimize all policies, handling discrete selections by score/enumeration).
