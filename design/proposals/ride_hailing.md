# Ride-hailing customer retention — continuous decisions and Bayesian priors

**Status:** future benchmark. Scalar continuous actions, mixed information, and joint policy-gradient training are implemented. The benchmark and its Bayesian-data extension remain unimplemented.

**Depends on.** The continuous-action version can use the current [decision-node and policy interfaces](../graph/decision_node.md). Scalar priors and conditioning on one observation also work. Fitting shared parameters from repeated observations and carrying their posterior uncertainty into later decisions remain separate future capabilities in [`TODO.md`](../../TODO.md).

## Where it comes from

The base problem is **Ride-Hailing Customer Retention** (`ferjorosa/decision-theory-llms`, `decision_problems/ride_hailing/problem.yaml`). A ride-hailing company decides whether to make a retention offer; the customer is `Good` or `Bad` (latent); the customer stays or leaves; the payoff depends on type and retention. It is small and discrete — which is exactly why it is the right base for both extensions.

## Extension 1 — continuous action, continuous outcome

The natural continuous move is to stop treating the offer as a coin flip and make it a *level*: a discount or retention incentive `x` (a real amount, or a fraction of the next fare). The action is then continuous, and retention and value become smooth functions of it.

```
type ~ Bernoulli(p_good)                             # latent Good/Bad
x ~ pi_theta(. | info)                              # stochastic incentive level, x in [0, 1]
stay | type, x ~ Bernoulli(σ(β0 + β1·x + β2·good))   # retention rises with the incentive
clv | type ~ LogNormal / Gamma                       # customer lifetime value, skewed
U = stay·clv − c(x)                                  # payoff net of the incentive
```

This combines a scalar continuous decision with categorical and continuous chance nodes. Two modeling choices matter:

- **Information set.** With no observed customer features the policy learns a parentless action distribution. An observed continuous feature, such as an app-engagement score, lets the distribution depend on that feature. `DistributionPolicy` with `Linear` can predict parameters for a distribution whose fixed support stays within `Interval(0, 1)`.
- **The discrete `stay` is downstream of the continuous action.** The current score-function engine handles sampled Bernoulli outcomes without differentiating through the sample. A conditional expected-payoff calculation can provide an independent reference for this small model. Continuous actions with categorical outcomes are covered in the [solver regression tests](../../tests/solver/test_continuous_policy.py).

Non-Gaussian is natural here: `LogNormal`/`Gamma` `clv`, and a `Beta` retention probability in Extension 2.

## Extension 2 — Bayesian priors

The base problem assumes known numbers: `P(good) = 0.9`, and known stay/leave probabilities. The Bayesian version treats those as unknown and learns them from data, which is where synthetic or real data enters.

- **What is uncertain.** The retention probabilities are the natural target: `stay_good ~ Beta(a, b)`, `stay_bad ~ Beta(a, b)`, optionally the type prior `p_good ~ Beta(…)`, with the base model's numbers as prior means. The decision then maximises posterior-predictive expected utility.
- **Data.** Synthetic: draw stay/leave counts from chosen rates and check the posterior recovers them — a self-contained test with a known answer. Real: a public churn/retention dataset, at the cost of mapping its columns onto `type` / `stay` / `clv`.
- **Why this problem.** Retention over a binary outcome is a Beta-Binomial model, the cleanest possible conjugate example, so the inference machinery can be validated independently of the decision layer.

This is Extension 1 with the parameters lifted to random variables. The two compose into a diagram with a continuous decision, continuous and discrete chance nodes, and uncertain model parameters. Sharing fitted parameter uncertainty between inference and later optimization remains a separate feature.

## Why this problem for continuous decisions

- The action is naturally a level (a discount), not a category, so it motivates a real-valued decision rather than a contrived one.
- The payoff has a smooth core (`P(stay | x)` is a logistic in `x`), which is the regime gradient-based policy optimization is designed for.
- It is small enough to compute an independent expected-utility reference over a grid of fixed actions.

## Validation

1. **Grid reference.** Evaluate expected utility directly over a fine grid of fixed actions. Compare the fitted stochastic policy with that reference, accounting for its distribution family, optimization error, and Monte Carlo evaluation noise.
2. **Closed forms.** For an LQG-shaped variant (linear-Gaussian chance, quadratic utility) the optimal affine policy is known; check the recovery.
3. **Monte Carlo checks.** Repeat training with several seeds and evaluate fitted policies on independent scenarios. Compare information sets without assuming monotone training progress or that each fitted policy attains its family's optimum.
4. **Bayesian check.** The posterior recovers synthetic rates, and the prior-to-posterior update matches the conjugate Beta-Binomial computed by hand.

## Depends on / deferred

- The benchmark still needs a model implementation and independent expected-utility references. Its scalar actions and numerical information can use the existing policy interfaces.
- Shared-parameter dataset fitting and decisions under learned uncertainty for Extension 2.
- The runnable benchmark remains deferred alongside the other example work in [`TODO.md`](../../TODO.md).

## Source

`https://github.com/ferjorosa/decision-theory-llms/blob/main/decision_problems/ride_hailing/problem.yaml`
