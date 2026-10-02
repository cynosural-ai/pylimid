# Documentation plan (beta)

Working document for the first published documentation site. Updated as the work progresses. Inspiration: [NumPyro](https://num.pyro.ai/en/stable/) for the layout (getting started → examples → API), [pyAgrum](https://pyagrum.readthedocs.io/en/3.1.1/) for notebooks-as-documentation.

## Status

| Step | Status |
| --- | --- |
| 0. Settle the public API | Not started |
| 1. Sphinx skeleton: home, quickstart, API reference, Read the Docs build | Not started |
| 2. Port the example notebooks + write the mixed influence-diagram example | Not started |
| 3. User guide pages | Not started |
| 4. Background pages + the pyAgrum solvability note | Not started |
| 5. Graphviz renderer, swapped into the examples | Not started |

Steps 0–2 are enough for a credible beta site; 3–5 can follow in later iterations.

## 0. Settle the public API first

A published API reference turns import paths into commitments. Before generating any API pages:

- **One front door for solving.** Today `decisionpy.inference.solve(diagram)` runs the slow unbatched scan and takes a diagram, while every example uses `batched_solve(diag.snapshot())`, which takes a snapshot. Proposal: `decisionpy.solve(diagram, method="auto" | "scan" | "backward_induction", num_samples=..., rng_key=...)`.
- **Top-level exports.** `decisionpy/__init__.py` exports only `__version__`. Re-export the main names (node types, `InfluenceDiagram`, `infer`, `solve`, the result types) so the quickstart imports from `decisionpy`.
- **Clean docstrings.** Remove internal wording that autodoc would publish verbatim (e.g. "Strategy B" in the `solve` docstrings).
- **Docstring convention vs. links.** `AGENTS.md` bans Sphinx roles and backticks in docstrings, so API pages would not cross-link (e.g. `Solution` to its class page). Decide whether to allow roles (or a `default_role`) in public docstrings. Napoleon handles the Google style either way.

This overlaps with "Reorganize the code" in `TODO.md`.

## 1. Site structure

1. **Home**: short pitch, a ~15-line code snippet, one figure, install command, a "what it is / what it isn't" box.
2. **Getting started**
   - Installation.
   - Quickstart: the medical-treatment example taken through build → `validate()` → `solve()` → `infer(..., policy=)`.
3. **User guide** (how this library works, not general theory)
   - Building a diagram: the mutable workspace, the three node types, consistency states (`UNCONFIGURED` / `STALE` / `CONSISTENT`), `validate()` / `snapshot()`.
   - Chance nodes: distributions as NumPyro callables; discrete, continuous, conditional linear-Gaussian (CLG).
   - Decisions and information sets: LIMID semantics (a decision sees exactly its parents, no implicit memory); memory arcs as a modelling choice.
   - Utilities: callables of the parents, multiple utilities add up, must be JAX-traceable.
   - Inference: `infer`, `Posterior` (`marginal` / `mean` / `hdi`), binding a policy.
   - Solving: which solver when, the solvability gate, Monte-Carlo error, `num_samples`, seeds.
   - Scope and limitations.
4. **Background** (short; each page links to the blog series for depth)
   - Decision analysis in one page.
   - From decision trees to influence diagrams to LIMIDs.
   - Solving influence diagrams: the scan, backward induction, solvability, Monte-Carlo decision analysis.
   - Technical note: the solvability criterion vs. pyAgrum's `isSolvable()`.
5. **Examples**: the notebooks.
6. **API reference**: autosummary pages, one per public module.
7. **References**: bibliography via `sphinxcontrib-bibtex`.
8. **Changelog**.

The ADRs and living notes in `docs/` stay out of the published site: they are written for contributors and full of history. A "Design notes" section can be added later if people ask.

Blog series to link from Background:

- [Decision theory I](https://ferjorosa.github.io/blog/2025/06/08/decision-theory-I.html): decision analysis, expected value, utility, decision trees (Oil Wildcatter).
- [Decision theory II](https://ferjorosa.github.io/blog/2025/07/04/decision-theory-II.html): limits of trees, influence diagrams, LIMIDs, arc reversal / node reduction.
- [Decision theory III](https://ferjorosa.github.io/blog/2025/08/07/decision-theory-III.html): the Oil LIMID in pyAgrum, sensitivity analysis, tornado diagrams.

## 2. Introduction and positioning

The first question a user will ask is "why not pyAgrum?". The home page and the scope page answer it plainly.

What is actually different:

- **Chance nodes are arbitrary NumPyro distributions**: discrete, continuous and mixed, not only CPTs or CLG.
- **Utilities are arbitrary functions**, including of continuous variables (`UtilityNode.values` is any JAX-traceable callable). What is tabular is the policy: the restriction is **discrete decisions with discrete information sets** (continuous decision parents are rejected in `solvers/_policy.py`).
- **LIMIDs from the start**: information sets are explicit, no implicit no-forgetting assumption.
- **Monte-Carlo solving** with a stated guarantee per solver, validated against pyAgrum's exact results.
- **A solvability gate that follows Lauritzen & Nilsson (2001) exactly**, stricter than pyAgrum's `isSolvable()`. The `D1 → X → D2` case in `inference/solver_algorithms.md` (pyAgrum returns expected utility 4.0, the optimum is 5.0) becomes a short technical note.

Be honest about when to use something else: for fully discrete diagrams that need exact answers, pyAgrum is the right tool. decisionpy is for continuous or mixed uncertainty with arbitrary distributions.

Draft opening line: *"decisionpy models and solves limited-memory influence diagrams whose chance variables can be discrete, continuous, or mixed, using NumPyro for probabilistic inference and Monte-Carlo solvers for the decisions."*

## 3. Examples (first iteration)

- **Medical treatment**: the quickstart (exists).
- **Oil Wildcatter**: the flagship. Connects to the blog series and already has a pyAgrum twin (`examples/03_influence_diagrams/_oil_wildcatter.py`). Both solvers next to the exact reference.
- **A mixed influence diagram** (missing, most important): e.g. the Oil Wildcatter with a continuous oil amount and price (lognormal) and a utility computed from them. Candidate may exist in the personal examples repo.
- **Policies and counterfactuals**: `counterfactuals_with_policy` (exists).
- **Bayesian networks**: one page grouping `categorical_bn` and `sangiovese`; secondary, shows `infer`.

## 4. Diagrams

Three different needs:

1. **Concept figures** (decision trees, ID vs. LIMID, arc-reversal steps): hand-made, reuse the blog assets. Keep the source files in `docs/_static/figures/`.
2. **Model pictures in examples**: add `to_dot()` + `_repr_svg_` with Graphviz, mirroring `graph/mermaid.py` (shapes per node kind, topological order, styling for dangling / stale nodes). Roughly a day of work; gives pyAgrum-style static SVGs that render in Jupyter, GitHub and Sphinx without JavaScript. Graphviz as an optional extra (`decisionpy[viz]`). Mermaid stays for text / LLM use.
   - Mermaid in Sphinx needs `sphinxcontrib-mermaid` and renders client-side; unclear whether Mermaid inside notebook Markdown outputs renders through myst-nb / nbsphinx without extra configuration.
3. **Results drawn inside nodes** (pyAgrum's posterior bars and policy tables in the graph): deferred. A matplotlib bar chart or a policy table in the notebook covers most of the value.

## 5. Tooling

- **Sphinx** + `myst-parser`: prose pages in Markdown.
- **`myst-nb`** for the examples: executes the jupytext percent `.py` files directly, so the `.py` twins stay the source of truth. `nb_execution_mode = "cache"`, or `"off"` with committed outputs if JAX on Read the Docs is too slow. (`nbsphinx`, used by NumPyro and pyAgrum, also works.)
- **API reference**: `autodoc` + `napoleon` + `autosummary`.
- **Extras**: `sphinxcontrib-bibtex`, `sphinx-copybutton`, `sphinx-design` (landing-page cards).
- **Theme**: Read the Docs theme (as in `TODO.md`, matches NumPyro); `pydata-sphinx-theme` or `furo` are a one-line switch later.
- **Hosting**: Read the Docs.
- **Release**: publish a pre-release (`0.1.0b1`) on PyPI so `pip install decisionpy` in the docs is real.

## 6. References

Grouped by where they are cited. Verify details before publishing.

- **Influence diagrams and their evaluation**
  - Howard, R. A. and Matheson, J. E. (1984). Influence diagrams.
  - Shachter, R. D. (1986). Evaluating influence diagrams. Operations Research.
  - Jensen, F., Jensen, F. V. and Dittmer, S. L. (1994). From influence diagrams to junction trees. UAI.
- **LIMIDs** (core)
  - Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science.
  - Nilsson, D. and Lauritzen, S. L. (2000). Evaluating influence diagrams using LIMIDs. UAI.
- **Continuous and mixed influence diagrams**
  - Shachter, R. D. and Kenley, C. R. (1989). Gaussian influence diagrams. Management Science.
  - Lauritzen, S. L. (1992). Propagation of probabilities, means and variances in mixed graphical association models. JASA.
  - Madsen, A. L. and Jensen, F. (2005). Solving linear-quadratic conditional Gaussian influence diagrams. IJAR.
  - Cobb, B. R. and Shenoy, P. P. (2008). Decision making with hybrid influence diagrams using mixtures of truncated exponentials. EJOR.
  - Li, Y. and Shenoy, P. P. (2010). Solving hybrid influence diagrams with deterministic variables. UAI.
- **Monte-Carlo decision analysis**
  - Bielza, C., Müller, P. and Ríos Insua, D. Decision analysis by augmented probability simulation. Management Science. (Likely 1999, 45(7); `inference/solver_algorithms.md` says 2007, 53(7) — verify.)
  - Charnes, J. M. and Shenoy, P. P. (2004). Multistage Monte Carlo method for solving influence diagrams using local computation. Management Science.
  - Kearns, M., Mansour, Y. and Ng, A. (1999). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. IJCAI.
  - Bellman, R. (1957). Dynamic Programming.
- **Textbooks**
  - Kjærulff, U. B. and Madsen, A. L. Bayesian Networks and Influence Diagrams.
  - Koller, D. and Friedman, N. Probabilistic Graphical Models (chapters 22–23).
  - Clemen, R. T. Making Hard Decisions.
- **Software**
  - Phan, D., Pradhan, N. and Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro.
  - Bingham, E. et al. (2019). Pyro: deep universal probabilistic programming. JMLR.
  - Ducamp, G., Gonzales, C. and Wuillemin, P.-H. (2020). aGrUM/pyAgrum: a toolbox to build models and algorithms for probabilistic graphical models in Python. PGM.

## Open decisions

- Shape of the public `solve()` (signature, `method=` names, what `"auto"` picks).
- Allow Sphinx roles in public docstrings or keep plain text.
- myst-nb vs. nbsphinx; execute on build vs. committed outputs.
- Theme.
