# Inference strategy for influence diagrams

This documents the reasoning behind the inference strategy: NumPyro as the only engine, with pyAgrum's exact solvers as the external validation reference.

---

## Current state

One engine — NumPyro — behind both public verbs:

- `infer()` — posterior inference. All-discrete diagrams use exact discrete enumeration (parallel enumeration of the discrete sites, bincounted into the probability vector); anything with a continuous latent uses NUTS with the discrete sites enumerated out. Results are `Posterior` objects (raw draws plus the variable's `states`), a Monte-Carlo estimate by contract.
- `solve()` — the intervention scan (Strategy B from `decision_node.md`): enumerate the discrete policy space, estimate each policy's expected utility by forward sampling with the decisions resolved from their observed information set, keep the best.

The exact engines (variable elimination, bucket elimination, the linear-Gaussian engine) were built and retired; see [`25_08_2026_numpyro_only_engine.md`](../ADR/25_08_2026_numpyro_only_engine.md) for the decision and `git log --all -- pylimid/inference/exact` for the history.

## Why NumPyro alone

The exact engines each added a result type, an extraction mechanism, and a precondition-checking path to the public API — the `exact` flag, the CPD probe, the result-type union. NumPyro alone covers the model classes in scope well:

- **Discrete nodes** — parallel enumeration gives exact-in-expectation posteriors without HMC.
- **Continuous nodes** — NUTS is auto-tuned and efficient, with no conjugacy constraints.
- **GPU/TPU** — free from JAX.
- **Mixed/continuous influence diagrams** — the intervention scan is the only solver that reaches them.

Exactness is not load-bearing anywhere in `infer()` or `solve()`; a Monte-Carlo estimate with the draws kept in hand is the honest contract. The one place exact answers still matter — validating the engine — is outsourced to pyAgrum: the comparison tests (`tests/inference/numpyro/test_compare_pyagrum*.py`) check `infer()` against pyAgrum's LazyPropagation, `solve()` against its exact LIMID solver, and the MCMC posteriors against `pyagrum.clg`.

## The road ahead

The deferred piece is **Strategy A — policy as parameters** (per `decision_node.md`): a decision backed by `numpyro.param()` forming a parameterized policy, with the utility pushed into the objective via `numpyro.factor()`, solved by gradient ascent on `E[U]` (reparameterization or score function). It is what continuous information sets and continuous decisions will require, and it composes with the existing scan for the mixed case (outer loop Strategy B, inner loop Strategy A).
