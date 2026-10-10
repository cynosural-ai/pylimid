# TODO

Forward plan. The engine work is done (numpyro-only, scan + batched scan + backward induction). The focus now is to finish the discrete-decision story before reaching for continuous decisions.

## 1. Finish discrete decisions

- **More worked examples in the docs.** Cover the decision classes the library is meant to demonstrate, so the guides and the API have something concrete to point at. The two planned problem benchmarks — the mixed continuous/discrete logistics center (buildable now) and the future continuous-decision ride-hailing problem — are scoped in `design/problems/`.
- **Proper documentation.** Turn the current living notes into reader-facing documentation: concepts, the solving strategies, and the result contract.
- **Bayesian priors as a feature.** A parameter of a distribution can itself be a chance node with a prior, which lets a model encode personal experience, such as the ice-cream vendor's belief about tomorrow's weather or their insight on demand. List it under Features in `docs/index.md` and `README.md`, and show it in the docs, for example as an extension of the quickstart. Things to know first:
  - Scalar priors already work in `solve` and `infer`: a `Beta` prior on the chance of sun, or a `LogNormal` scale on the demand mean, solves and updates correctly after an observation.
  - Vector-valued priors do not: a `Dirichlet` node over the weather probabilities solves, but `infer` crashes when it builds the `Posterior`, because `_infer_numpyro` in `pylimid/inference/engine.py` converts every draw with `float()`. Fix that before advertising Dirichlet priors (see Known issues).
  - Pick the example carefully. A prior on the weather probabilities alone leaves the quickstart's policy unchanged, since only its mean matters for a single forecast. A prior on demand does change it (a cloudy-day stock of 80 instead of 60), so it is the more convincing demonstration.
- **Reorganize the code.** Settle the module layout now, with discrete decisions as the scope, so later work lands on a stable surface.

## 2. Library documentation site

Give the project a published documentation page, the way pyAgrum and NumPyro have: a proper site with guides and an API reference, rather than the development notes in `design/`, which are aimed at contributors.

Built with [Sphinx](https://www.sphinx-doc.org/) using the [Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io/) and hosted on Read the Docs.

## 3. Continuous decisions (afterwards)

Only once 1 and 2 are done. Strategy A — continuous actions and continuous information sets — is the likely next feature. The exploration map (feasibility, difficulty axes, guarantee change, validation plan, scope ladder, and the revisit trigger) lives in `design/continuous_decisions/` (on the unmerged `docs/continuous-decisions` branch, still under the old `docs/` path).

## Known issues

Found while writing the user guide. Ordered by how badly they can mislead a user.

### Wrong answers without an error

- **A discrete node's distribution can produce values outside its states.** Nothing checks that a discrete node only samples `0` to `len(states) - 1`. A node with six `states` and a `Poisson` distribution passes `validate()`, and `solve()` returns a policy in which every draw of 6 or more is treated as state 5, because JAX clamps out-of-range indices silently. About 8% of draws were affected in the test. Possible fix: have `probe_discrete_parents()`, or a new check, sample each discrete node and report values outside its states.
- **Chained table indexing (`T[a][b]`) under `infer()`.** This is a NumPyro bug, [pyro-ppl/numpyro#2252](https://github.com/pyro-ppl/numpyro/issues/2252), and the maintainers say the chained form cannot be made to work. With a continuous latent (the NUTS path) the posterior is silently wrong: 0.53 against an exact 0.22 in the test. With only discrete latents it crashes with `KeyError: '_pyro_dim_…'`. `solve()` is not affected. What we can do:
  - Catch the `KeyError` and re-raise it with a hint to index in one step, `T[a, b]`.
  - Investigate a check that calls each `dist` with parent values carrying an extra leading dimension and verifies the output shape; this might catch the silent case. Untested idea.
  - When the upstream PR that improves the `KeyError` lands, revisit the note in `design/inference/backend_numpyro.md`.

### Crashes with unhelpful messages

- **Vector-valued nodes crash `infer()`.** A `Dirichlet` node, or any node whose draws are arrays, fails because `_infer_numpyro` in `pylimid/inference/engine.py` converts every draw with `float()`.
- **`IndexError: list index out of range` from `infer()`** when a discrete node samples beyond its states (the `Poisson` case above, with no observations). The message does not name the cause. Fixing the states check above mostly covers this.

### Cosmetic

- **Doubled quote in messages.** `{name!r}'s` renders as `'forecast''s` because `!r` already adds quotes. It appears in `pylimid/graph/diagram.py` (the probe), `chance_node.py` and `utility_node.py` (the signature checks).

### Missing features

- **`infer()` has no `num_samples` or `rng_key`.** It always draws 2,000 samples (with 500 warmup steps under NUTS) from `PRNGKey(0)`, while `solve()` accepts both.
- **Every `solve()` and `infer()` call compiles again**, because the jitted functions are defined inside each call. That costs about 0.3–0.6 s even on the quickstart model, independent of `num_samples`. Caching the compiled functions would help anyone who calls them in a loop.
