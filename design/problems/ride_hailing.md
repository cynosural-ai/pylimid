# Ride-hailing customer retention — continuous decisions and Bayesian priors

**Status:** future. Two extensions parked behind the discrete-decision roadmap. This note is the problem statement they will be built against.

**Depends on.** Continuous decisions and continuous utilities are the policy-optimization (Strategy A) work mapped in the continuous-decisions notes (currently `docs/continuous_decisions/` on the unmerged `docs/continuous-decisions` branch; to move to `design/continuous_decisions/`). Bayesian priors over the model parameters depend on the deferred parametric-learning / `fit` story ([`README.md`](../README.md), status table). Neither exists yet.

## Where it comes from

The base problem is **Ride-Hailing Customer Retention** (`ferjorosa/decision-theory-llms`, `decision_problems/ride_hailing/problem.yaml`). A ride-hailing company decides whether to make a retention offer; the customer is `Good` or `Bad` (latent); the customer stays or leaves; the payoff depends on type and retention. It is small and discrete — which is exactly why it is the right base for both extensions.

## Extension 1 — continuous action, continuous outcome

The natural continuous move is to stop treating the offer as a coin flip and make it a *level*: a discount or retention incentive `x` (a real amount, or a fraction of the next fare). The action is then continuous, and retention and value become smooth functions of it.

```
type ~ Bernoulli(p_good)                             # latent Good/Bad
x = δ(info)                                          # decision: incentive level, x in [0, 1]
stay | type, x ~ Bernoulli(σ(β0 + β1·x + β2·good))   # retention rises with the incentive
clv | type ~ LogNormal / Gamma                       # customer lifetime value, skewed
U = stay·clv − c(x)                                  # payoff net of the incentive
```

This is the "continuous decision + mixed chance nodes" class of the strategy map. Two design forks matter:

- **Information set.** With no observed customer features the policy is a single scalar `δ() = x`. Give the decision an observed continuous feature (an app-engagement score, say) and the policy becomes a function `δ(feature)` — the continuous-information-set case, and the more interesting benchmark. Affine policies are the first family in the continuous-decisions map.
- **The discrete `stay` is downstream of the continuous action.** This is the map's caveat case: sampling a Bernoulli whose probability is `σ(β0 + β1·x + …)` gives a zero pathwise gradient through `x`. Three ways out, in increasing generality: (a) replace the sampled `stay` with its smooth expected value `P(stay | type, x)` inside the utility — retention as a probability weight rather than a draw; (b) enumerate the Bernoulli exactly inside the objective (cheap, it is binary); (c) use a score-function estimator. (a) is the cleanest first benchmark; (b) and (c) exercise the genuinely mixed machinery. Making retention continuous instead — retained months, or retained revenue — avoids the issue altogether and is the safe all-continuous variant.

Non-Gaussian is natural here: `LogNormal`/`Gamma` `clv`, and a `Beta` retention probability in Extension 2.

## Extension 2 — Bayesian priors

The base problem assumes known numbers: `P(good) = 0.9`, and known stay/leave probabilities. The Bayesian version treats those as unknown and learns them from data, which is where synthetic or real data enters.

- **What is uncertain.** The retention probabilities are the natural target: `stay_good ~ Beta(a, b)`, `stay_bad ~ Beta(a, b)`, optionally the type prior `p_good ~ Beta(…)`, with the base model's numbers as prior means. The decision then maximises posterior-predictive expected utility.
- **Data.** Synthetic: draw stay/leave counts from chosen rates and check the posterior recovers them — a self-contained test with a known answer. Real: a public churn/retention dataset, at the cost of mapping its columns onto `type` / `stay` / `clv`.
- **Why this problem.** Retention over a binary outcome is a Beta-Binomial model, the cleanest possible conjugate example, so the inference machinery can be validated independently of the decision layer.

This is Extension 1 with the parameters lifted to random variables. The two compose into a diagram with a continuous decision, continuous and discrete chance nodes, and hierarchical parameters — close to the hardest class in the strategy map, and therefore a good *last* benchmark rather than a first one.

## Why this problem for continuous decisions

- The action is naturally a level (a discount), not a category, so it motivates a real-valued decision rather than a contrived one.
- The payoff has a smooth core (`P(stay | x)` is a logistic in `x`), which is the regime gradient-based policy optimization is designed for.
- It is small enough to validate against a grid-discretized action solved with the existing scan.

## Validation

1. **Grid fallback.** Discretize the continuous action onto a grid, solve with the existing scan, and check the continuous optimum approaches the grid optimum as the grid refines, within discretization bias.
2. **Closed forms.** For an LQG-shaped variant (linear-Gaussian chance, quadratic utility) the optimal affine policy is known; check the recovery.
3. **Monte-Carlo invariants.** Common random numbers, monotone improvement across optimizer steps, seed stability, and `no info ≤ noisy info ≤ full info`.
4. **Bayesian check.** The posterior recovers synthetic rates, and the prior-to-posterior update matches the conjugate Beta-Binomial computed by hand.

## Depends on / deferred

- Continuous decision representation and parameterized policies; a continuous policy is a function, not a table, so the result contract changes.
- Continuous information sets, if the observed-feature fork is taken.
- Parametric learning / `fit` for Extension 2.
- All of it is parked until the discrete-decision work in [`TODO.md`](../../TODO.md) is finished.

## Source

`https://github.com/ferjorosa/decision-theory-llms/blob/main/decision_problems/ride_hailing/problem.yaml`
