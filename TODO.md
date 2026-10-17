# TODO

Forward plan. The engine work is done (numpyro-only, scan + batched scan + backward induction). The focus now is to finish the discrete-decision story before reaching for continuous decisions.

## 1. Finish discrete decisions

- **Bayesian priors as a feature.** A parameter of a distribution can itself be a chance node with a prior, which lets a model encode personal experience, such as the ice cream shop owner's belief about tomorrow's weather or their insight on demand. List it under Features in `docs/index.md` and `README.md`, and show it in the docs, for example as an extension of the quickstart. Things to know first:
  - Scalar priors already work in `solve` and `infer`: a `Beta` prior on the chance of sun, or a `LogNormal` scale on the demand mean, solves and updates correctly after an observation.
  - Pick the example carefully. A prior on the weather probabilities alone leaves the quickstart's policy unchanged, since only its mean matters for a single forecast. A prior on demand does change it (a cloudy-day stock of 80 instead of 60), so it is the more convincing demonstration.

## 3. Continuous decisions (afterwards)

Only once 1 and 2 are done. Strategy A — continuous actions and continuous information sets — is the likely next feature. The exploration map (feasibility, difficulty axes, guarantee change, validation plan, scope ladder, and the revisit trigger) lives in `design/continuous_decisions/` (on the unmerged `docs/continuous-decisions` branch, still under the old `docs/` path).

## Tutorial extensions (later)

- **Logistics-center consequences.** The mixed continuous-score model is implemented in [`docs/tutorials/logistics_center.md`](docs/tutorials/logistics_center.md), with exact arithmetic on its collapsed discrete model as the reference. A future extension could add continuous downtime or revenue feeding the payoff. Calibrating the report probabilities to the original discrete problem is optional further work; the current Gamma example uses its own parameterization. A Poisson failure-count extension would also need numerical discrete-node support.

## Known issues

Found while writing the user guide. Ordered by how badly they can mislead a user.

### Wrong answers without an error

- **Chained table indexing (`T[a][b]`) under `infer()`.** This is a NumPyro bug, [pyro-ppl/numpyro#2252](https://github.com/pyro-ppl/numpyro/issues/2252), and the maintainers say the chained form cannot be made to work. With a continuous latent (the NUTS path) the posterior is silently wrong: 0.53 against an exact 0.22 in the test. With only discrete latents it crashes with `KeyError: '_pyro_dim_…'`. `solve()` is not affected. What we can do:
  - Catch the `KeyError` and re-raise it with a hint to index in one step, `T[a, b]`.
  - Investigate a check that calls each `dist` with parent values carrying an extra leading dimension and verifies the output shape; this might catch the silent case. Untested idea.
  - When the upstream PR that improves the `KeyError` lands, revisit the note in `design/inference/backend_numpyro.md`.

### Situations that never come up in the simulation

A situation is one combination of a decision's parent states. When no sample lands in one, the solver can't tell whether it is rare (it can happen, but the simulation missed it) or impossible (its probability is 0, such as "test not run, result positive").

- **Backward induction raises on every unseen situation, including impossible ones.** This blocks valid models: the common `no_results` idiom, a state that is certain when a test isn't run, can't be solved with backward induction. Proposal: finish the solve, give unseen situations an arbitrary action, and warn with the list of them, so the user can ignore impossible ones and re-solve with more samples for rare ones. If a warning is too easy to miss, add a parameter and keep raising by default.
- **The scan gives unseen situations an arbitrary action without saying so.** Harmless for impossible situations, but a rare one that the simulation missed also gets an arbitrary action, and nothing tells the user. It should warn like backward induction would.
- **The scan repeats work for impossible situations.** Policies that differ only in the action for an impossible situation get identical scores, so each impossible situation multiplies the number of policies by the decision's number of actions (the `no_results` idiom in the oil field would mean 2³ = 8 times more policies for `buy`). Possible fix: a short pilot simulation finds the situations that never come up, they get one fixed action, and the scan enumerates only the rest.

### Documentation

- **Learning from data (future guide).** Introduce uncertainty about model parameters and updating those beliefs from observations separately from the inference guide. A scalar prior such as `busyness` and one observation work already. Repeated datasets and decisions under learned uncertainty can extend the page once those capabilities are available.
- **Remake the oil field tutorial.** Its note on the result when the test isn't run says that with a `no_results` state "`solve()` would fall back to the scan". It doesn't: `auto` picks the solver from the structure alone (`is_solvable`), the diagram is solvable, so backward induction runs and raises, and there is no fallback after an error. The tutorial avoids the case with a 50/50 filler for `result`; once backward induction warns instead of raising, the `no_results` idiom would work as it is.

### Missing features

- **Numerical discrete chance nodes (future proposal).** Chance nodes currently support labeled Categorical and Bernoulli distributions; runtime validation rejects incompatible distributions or category counts before sampling. Binomial and Poisson remain unsupported, with or without labels. The [discrete chance-node proposal](design/graph/discrete_chance_nodes.md) covers numerical count semantics, finite domains for policy tables, and operation-specific inference requirements. The public API is still to be agreed.
- **Sampler diagnostics and controls.** Expose configurable NUTS warmup through `infer` and return diagnostics that help users assess sampling reliability, including divergences and effective sample sizes. Support multiple chains for convergence checks and report R-hat only when applicable. Forward sampling and discrete enumeration should distinguish diagnostics that do not apply.
- **Dataset fitting (later).** Define how repeated observations share model parameters, rather than relying on broadcasting alone. A fitting tutorial should cover learning shared parameters and predicting new observations; keep this out of the inference-guide rewrite.
- **Decisions under learned uncertainty (later).** Provide a way to carry posterior uncertainty from `infer` into future simulations used by `solve`, preserving dependencies between learned parameters. This is separate from accepting datasets; plugging in posterior means alone would discard that uncertainty.
- **Every `solve()` and `infer()` call compiles again**, because the jitted functions are defined inside each call. That costs about 0.3–0.6 s even on the quickstart model, independent of `num_samples`. Caching the compiled functions would help anyone who calls them in a loop.


------

### Comments from Fer

- Should we have a 3rd guide about "Fitting a model"?
- After we have defined all of the main pages, reviewed the API, have good tutorials, etc. we would take a look into LLM-based integration, most probably with a different repo. And we would decide then how to document it
