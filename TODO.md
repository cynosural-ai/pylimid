# TODO

Forward plan. The engine work is done (numpyro-only, scan + batched scan + backward induction). The focus now is to finish the discrete-decision story before reaching for continuous decisions.

## 1. Finish discrete decisions

- **More examples.** Cover the decision classes the library is meant to demonstrate, so the docs and the API have something concrete to point at. The two planned problem benchmarks — the mixed continuous/discrete logistics center (buildable now) and the future continuous-decision ride-hailing problem — are scoped in `design/problems/`.
- **Proper documentation.** Turn the current living notes into reader-facing documentation: concepts, the solving strategies, and the result contract.
- **Reorganize the code.** Settle the module layout now, with discrete decisions as the scope, so later work lands on a stable surface.

## 2. Library documentation site

Give the project a published documentation page, the way pyAgrum and NumPyro have: a proper site with guides and an API reference, rather than the development notes in `design/`, which are aimed at contributors.

Built with [Sphinx](https://www.sphinx-doc.org/) using the [Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io/) and hosted on Read the Docs.

## 3. Continuous decisions (afterwards)

Only once 1 and 2 are done. Strategy A — continuous actions and continuous information sets — is the likely next feature. The exploration map (feasibility, difficulty axes, guarantee change, validation plan, scope ladder, and the revisit trigger) lives in `design/continuous_decisions/` (on the unmerged `docs/continuous-decisions` branch, still under the old `docs/` path).
