# Automated logistics center — mixed continuous/discrete chance nodes

**Status:** implemented — first cut. The shared model lives in [`examples/03_influence_diagrams/_logistics_center.py`](../../examples/03_influence_diagrams/_logistics_center.py), the example walkthrough in [`examples/03_influence_diagrams/mixed_logistics_center.ipynb`](../../examples/03_influence_diagrams/mixed_logistics_center.ipynb), and the published tutorial in [`docs/tutorials/logistics_center.md`](../../docs/tutorials/logistics_center.md). The model uses the continuous-score variant (1) below with a fresh Gamma parameterization and validates against pyAgrum solving the collapsed discrete model; the continuous-consequence variant (2) and exact table calibration are still open. This section records the design the implementation follows.

**Scope.** A single-problem design note. The library behavior it rests on — continuous chance nodes, discrete decisions and utilities, the NumPyro solver family — is settled elsewhere ([`chance_node.md`](../graph/chance_node.md), [`decision_node.md`](../graph/decision_node.md), [`backend_numpyro.md`](../inference/backend_numpyro.md)). The case where a *decision observes a continuous variable* is out of scope here; that is a solver-generalization question, not a problem-design one.

## Where it comes from

The base problem is the **Automated Logistics Center** decision problem (`ferjorosa/decision-theory-llms`, `decision_problems/automated_logistics_center/problem.yaml`). It is a fully discrete influence diagram — a test decision, an investment decision, discrete system states, and tabular utilities in € millions. It is the right base because its decisions are already discrete, so adding continuous *chance* nodes never collides with the solver's discrete-information-set requirement (`info_assignments`, `pylimid/inference/numpyro/solvers/_policy.py`). The other candidate, ride-hailing, is better suited to the continuous-decision case (see [`ride_hailing.md`](./ride_hailing.md)).

## The base model

- `T` — decision: `test` / `no test` / `do nothing`.
- `A` — chance: advanced system state, `smooth` / `minor failure` / `major failure`.
- `C` — chance: conventional system state, `smooth` / `failure`.
- `R` — chance: test report, `bad` / `good` / `excellent`, a child of `A`.
- `I` — decision: invest `advanced` / `conventional`; its information set is `R` when testing, empty otherwise.
- `U` — utility: a tabular payoff per branch, in € million, including the €10M test cost on the tested branches.

Everything here is finite and exact, and it already has an exact external oracle (pyAgrum's `ShaferShenoyLIMIDInference`, as in [`_limid_fixtures.py`](../../tests/inference/numpyro/_limid_fixtures.py)). That oracle is what makes the mixed extension testable: keep the discrete version as the reference.

## What we add

Two independent ways to introduce continuous chance nodes. Neither puts a continuous variable in a decision's information set, so the current scan and backward-induction solvers keep working.

### 1. A continuous latent score, binned into the existing report

Replace the directly enumerated `P(R | A)` with a latent continuous score for the advanced system:

```
s | A ~ p_A          # continuous, e.g. StudentT / Gamma / LogNormal / Weibull
R = bin(s)           # the existing bad / good / excellent report
```

`R` stays discrete, so `I`'s information set stays discrete and the solver is untouched. The report's probabilities are now *induced* by a continuous density and its thresholds rather than written down directly. Pick a non-Gaussian `p_A` on purpose — this is the point of the exercise. A `Gamma`/`LogNormal` score is positive and skewed (a natural model for a performance or time-to-failure measurement); a `StudentT` score is heavy-tailed (a test that occasionally throws large outliers in the same units as the state).

Calibrating the thresholds so the induced `P(R | A)` matches the original table would give a built-in regression to the discrete baseline, but it is over-constrained for a single natural family and shared thresholds. The implementation instead uses a fresh parameterization and checks the mixed model against pyAgrum solving the *collapsed* discrete model built from the induced `P(R | A)` — the same expectation, reached through the exact solver.

### 2. A continuous consequence feeding the payoff

Add a continuous outcome downstream of the decisions, consumed only by the utility:

```
F | A ~ Poisson(λ_A)        # failures in the first year
D | F ~ Gamma(...)          # downtime given the failure count
U  = base(...) − penalty(D)
```

or a demand/revenue term `Demand ~ LogNormal` that scales the chosen system's payoff. These nodes sit off every decision's information set and off the differentiated path (there are no continuous decisions in this problem), so the solvers handle them by Monte Carlo over the sampled values — exactly as they already handle any continuous `dist` callable.

**Recommended first cut.** Add only (1). It is the smallest change that makes the diagram mixed, and it is the change for which the discrete model is an exact fallback. Add (2) as a follow-up once (1) is validated end to end.

## Why this problem

- Decisions stay discrete, so the mixed extension does not require the not-yet-implemented continuous-decision or continuous-information-set machinery.
- The existing utility tables survive the first cut unchanged; the discrete diagram is the reference answer.
- It exercises non-Gaussian continuous sampling, the `Posterior` continuous methods (`mean()` / `std()` / `hdi()`), and forward/posterior inference on a real (not toy) diagram.
- Value-of-information questions come for free: how much does the continuous test reduce uncertainty about `A`, and how does that move the investment decision?

## pylimid sketch

- `A`, `C` as today (discrete root chances).
- `s` — continuous child of `A`, `states=None`, a `dist` callable returning the chosen non-Gaussian family.
- `R` — discrete child of `s`, a `Categorical` whose probabilities come from the bin thresholds (or a deterministic `bin(s)`), rather than a direct child of `A`.
- `T`, `I` unchanged — discrete decisions, discrete information sets.
- `U` unchanged in the first cut; extended to consume `D` / `Demand` in the follow-up.

## Validation

1. **Collapsed-discrete reference.** Build the all-discrete diagram whose `P(R | A)` is the one the continuous score induces, solve it exactly with pyAgrum, and require the mixed model's MEU and the test-branch policy to agree up to Monte-Carlo noise (`_logistics_center.py` does exactly this).
2. **Monte-Carlo convergence.** MEU and the continuous marginals stabilise as the sample count grows; the scan already reports Monte-Carlo estimates.
3. **Value-of-information sanity.** A more informative `s` (narrower per-state densities) cannot hurt: `MEU(test) − MEU(no test)` is non-negative and increases with informativeness.
4. **Continuous posterior.** `infer()` on `s` given an observed `R` returns draws whose `mean()` / `std()` / `hdi()` respond correctly to the bin.

## Deferred

- The decision observing `s` directly, rather than its binned report. That is a continuous information set — the solver-generalization work in the continuous-decisions map, not a change to this problem.
- Continuous decisions and continuous utilities; those belong to [`ride_hailing.md`](./ride_hailing.md).

## Source

`https://github.com/ferjorosa/decision-theory-llms/blob/main/decision_problems/automated_logistics_center/problem.yaml`
