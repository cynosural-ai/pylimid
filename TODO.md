# TODO

Forward plan. The engine work is done (numpyro-only, scan + batched scan + backward induction). The focus now is to finish the discrete-decision story before reaching for continuous decisions.

## 1. Finish discrete decisions

- **More worked examples in the docs.** Cover the decision classes the library is meant to demonstrate, so the guides and the API have something concrete to point at. The two planned problem benchmarks — the mixed continuous/discrete logistics center (buildable now) and the future continuous-decision ride-hailing problem — are scoped in `design/problems/`.
- **Proper documentation.** Turn the current living notes into reader-facing documentation: concepts, the solving strategies, and the result contract.
- **Bayesian priors as a feature.** A parameter of a distribution can itself be a chance node with a prior, which lets a model encode personal experience, such as the ice cream shop owner's belief about tomorrow's weather or their insight on demand. List it under Features in `docs/index.md` and `README.md`, and show it in the docs, for example as an extension of the quickstart. Things to know first:
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

### Situations that never come up in the simulation

A situation is one combination of a decision's parent states. When no sample lands in one, the solver can't tell whether it is rare (it can happen, but the simulation missed it) or impossible (its probability is 0, such as "test not run, result positive").

- **Backward induction raises on every unseen situation, including impossible ones.** This blocks valid models: the common `no_results` idiom, a state that is certain when a test isn't run, can't be solved with backward induction. The error also says to increase `num_samples`, which can never help for an impossible situation. Proposal: finish the solve, give unseen situations an arbitrary action, and warn with the list of them, so the user can ignore impossible ones and re-solve with more samples for rare ones. If a warning is too easy to miss, add a parameter and keep raising by default. Either way, the message should say that more samples won't help if the situation is impossible.
- **The scan gives unseen situations an arbitrary action without saying so.** Harmless for impossible situations, but a rare one that the simulation missed also gets an arbitrary action, and nothing tells the user. It should warn like backward induction would.
- **The scan repeats work for impossible situations.** Policies that differ only in the action for an impossible situation get identical scores, so each impossible situation multiplies the number of policies by the decision's number of actions (the `no_results` idiom in the oil field would mean 2³ = 8 times more policies for `buy`). Possible fix: a short pilot simulation finds the situations that never come up, they get one fixed action, and the scan enumerates only the rest.

### Cosmetic

- **Doubled quote in messages.** `{name!r}'s` renders as `'forecast''s` because `!r` already adds quotes. It appears in `pylimid/graph/diagram.py` (the probe), `chance_node.py` and `utility_node.py` (the signature checks).
- **Stale solver name in an error.** `solve(method="backward_induction")` on an unsolvable diagram says "Use the intervention scan"; the method is called `scan`.

### Documentation

- **Remake the oil field tutorial.** Its note on the result when the test isn't run says that with a `no_results` state "`solve()` would fall back to the scan". It doesn't: `auto` picks the solver from the structure alone (`is_solvable`), the diagram is solvable, so backward induction runs and raises, and there is no fallback after an error. The tutorial avoids the case with a 50/50 filler for `result`; once backward induction warns instead of raising, the `no_results` idiom would work as it is.

### Missing features

- **`infer()` can't query a utility.** The natural "if I always stock 100, how often do I lose money?" question (a risk profile) has no direct answer. A chance node with a `Delta` distribution works as a workaround when there is no evidence, since that runs forward sampling, but NUTS rejects it as soon as a chance node is observed.
- **Every `solve()` and `infer()` call compiles again**, because the jitted functions are defined inside each call. That costs about 0.3–0.6 s even on the quickstart model, independent of `num_samples`. Caching the compiled functions would help anyone who calls them in a loop.


------

### Comments from Fer

- Should we have a 3rd guide about "Fitting a model"?
- Should we have a user guide about infer? (i would say so)
- After we have defined all of the main pages, reviewed the API, have good tutorials, etc. we would take a look into LLM-based integration, most probably with a different repo. And we would decide then how to document it