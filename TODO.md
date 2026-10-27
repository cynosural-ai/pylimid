# TODO

PyLIMID focuses on flexible policy-gradient optimization for mixed influence diagrams. The earlier table-solving engines and pyAgrum comparison suites have been removed from this branch.

## Policy-gradient engine

`solve` jointly fits `LinearSoftmax` and `TabularSoftmax` rules using declared local information and evaluates the final stochastic policy on independent scenarios. There is no solver selection. See the [optimizer design](design/solver/algorithms.md) for the implementation boundary and analytic verification.

Next work includes a common trainable-family interface, custom neural families, richer baselines, checkpoint selection, and convergence diagnostics. Continuous actions remain separate.

## Documentation rewrite

The published guides, README examples, and API reference still describe the previous solver API and await a separate rewrite. They are intentionally not migrated as part of the engine refactor.

## Continuous decisions (later)

Real-valued actions remain a separate extension, including action domains and suitable gradient estimators. The broader exploration is on the unmerged [`docs/continuous-decisions` draft](https://github.com/cynosural-ai/pylimid/pull/41), under `design/continuous_decisions/`. Its original grouping of continuous actions and continuous information is split by the proposal above; accepting continuous observations for discrete actions does not require continuous-action support.

## Tutorial extensions (later)

- **Logistics-center consequences.** The mixed continuous-score model is implemented in [`docs/tutorials/logistics_center.md`](docs/tutorials/logistics_center.md), with exact arithmetic on its collapsed discrete model as the reference. A future extension could add continuous downtime or revenue feeding the payoff. Calibrating the report probabilities to the original discrete problem is optional further work; the current Gamma example uses its own parameterization. A Poisson failure-count extension would also need numerical discrete-node support.

## Known issues

Found while writing the user guide. Ordered by how badly they can mislead a user.

### Wrong answers without an error

- **Chained table indexing (`T[a][b]`) under `infer()`.** This is a NumPyro bug, [pyro-ppl/numpyro#2252](https://github.com/pyro-ppl/numpyro/issues/2252), and the maintainers say the chained form cannot be made to work. With a continuous latent (the NUTS path) the posterior is silently wrong: 0.53 against an exact 0.22 in the test. With only discrete latents it crashes with `KeyError: '_pyro_dim_…'`. `solve()` is not affected. What we can do:
  - Catch the `KeyError` and re-raise it with a hint to index in one step, `T[a, b]`.
  - Investigate a check that calls each `dist` with parent values carrying an extra leading dimension and verifies the output shape; this might catch the silent case. Untested idea.
  - When the upstream PR that improves the `KeyError` lands, revisit the note in `design/inference/backend_numpyro.md`.

### Documentation

- **Learning from data (future guide).** Introduce uncertainty about model parameters and updating those beliefs from observations separately from the inference guide. A scalar prior such as `busyness` and one observation work already. Use fully observed demand, including unmet requests, to update `busyness`; sales alone do not reveal demand after a stockout. Repeated datasets and decisions under learned uncertainty can extend the page once those capabilities are available.

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
