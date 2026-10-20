# Documentation maintenance

Maintenance guide for the published documentation in `docs/` and the project `README.md`. The first documentation version, including the five core user guides, is complete. Read this before changing published pages, and update it when the structure or conventions change. Public API details belong in docstrings and the API reference; build instructions live in [docs/README.md](../docs/README.md).

## Site structure and scope

The published order is **Getting started → User guide → Resources → API reference**. Tutorials will sit between User guide and Resources when ready.

- **Getting started** covers installation and a quickstart: one small problem through build, validate, draw and solve.
- **User guide** answers how to do something and explains the behaviour where readers encounter it. Its five core pages cover building a diagram, distributions and utilities, solving, inference, and multiple decisions.
- **Tutorials** tell the story of a complete alternative problem, with figures and interpretation of the results.
- **Resources** contains Common pitfalls and Count distributions, directly under the sidebar heading. Entries are standalone examples with their own context and imports. Count distributions collects count-specific examples and limitations. Background and further reading can be added when useful.
- **API reference** documents public classes, functions and methods, generated from docstrings.

Contributor documentation belongs in `CONTRIBUTING.md`, `AGENTS.md` and `design/`. Explain architecture on the published site only when it helps users understand execution or trust a result. Link to NumPyro for distributions and MCMC; focus PyLIMID explanations on information sets, explicit memory, policies, solver selection, callable parents and decision-modelling patterns. Introduce theory beside the behaviour it explains, with a citation rather than a textbook derivation. Preserve the Solving guide's explanation of the stricter solvability check; its rationale and regression example live in the [solver design](solver/algorithms.md).

### Shared example

The quickstart and user guide follow the ice cream shop. The guides extend the quickstart with hotel bookings; the Multiple decisions guide adds a promotion decision. The Count distributions page uses standalone examples so it does not interrupt this story. Keep the shared model consistent across pages while preserving intentional variants, such as revealing bookings before ordering in the inference guide.

Keep the quickstart and README's order menu at 0–200. The Solving guide deliberately starts with the same menu, then expands it to 400 to teach readers to check boundary answers. Preserve this progression and re-check the results whenever the model changes. Model parameters and numerical outputs live in the pages and their figure scripts.

## Writing conventions

### Voice and terminology

- Write complete sentences for readers new to the library. Introduce a concrete problem before code and explain terms at first use. Prefer plain words and active voice where the actor matters.
- Use “we” for the author and reader building models and interpreting results. Use “the owner” or “the shop” for the example's decisions, avoiding pronouns for the owner. Introduce scenarios directly, for example “Suppose today is sunny”.
- Use present tense for walkthrough steps and API behaviour; past tense for earlier steps and conditionals for hypothetical alternatives.
- Keep pages short by choosing what to include. Keep explanations, code and output that help readers understand the example; put reusable debugging examples in Common pitfalls.
- Write **PyLIMID** in prose and `pylimid` for the package, imports, extras and URLs. Use **node** for diagram structure and **variable** for the represented value. A diagram can combine discrete and continuous variables; a single variable is not “mixed”.
- Format specific node names, Python identifiers and literal state labels as code; keep ordinary concepts in plain text.
- Let Markdown lines flow; do not wrap at 80 characters.

### Accuracy and examples

- Check API claims against the implementation and API reference. Distinguish table solvers, which require categorical information sets, from explicit single-decision `policy_gradient` training of `LinearSoftmax`. Supplied deterministic functions and fitted stochastic rules can use continuous information during forward evaluation; posterior inference through these rules remains unsupported. Describe policy-gradient results as approximate candidates with independent utility evaluation, not certified optima. Describe supported chance distributions and state labels accurately; keep the full contract in `ChanceNode`.
- Distributions and utilities must be JAX-traceable. Do not describe them as arbitrary Python code; explain the restrictions on casts and Python control flow where relevant.
- Both `infer` and `solve` return Monte Carlo estimates. Distinguish exact discrete enumeration as a method from the accuracy of the returned summaries, including in mixed models. Explain sampling reliability where readers use estimates. When documenting unsampled situations, distinguish impossible combinations from rare ones, and explain the warning and unoptimised default actions.
- Keep performance claims modest. Explain when pyAgrum's exact discrete solving may be a better choice.
- Every number quoted in prose must come from running the page's code. Re-run and check prose and figures when a model changes. Check whether an optimal action lies at the edge of its menu, and explain intentional boundary examples.
- Pages show code that teaches PyLIMID. Generate plots separately, store their scripts beside the images in `docs/_static/figures/<page>/`, and embed images with `{figure}` directives. Keep each script's parameters consistent with its page.
- Recommend the `cuda12` and `cuda13` installation extras; the `gpu` extra is deliberately not advertised.

### Links and API documentation

- Use relative links between pages in `docs/`. In `README.md`, link to the rendered Read the Docs pages so readers see executed outputs.
- Link the first API-name mention in each section using MyST roles, such as ``{py:class}`InfluenceDiagram` ``, ``{py:meth}`~InfluenceDiagram.validate` `` and ``{py:func}`solve` ``. Do not link inside headings or code cells.
- Resolve named Python, JAX and NumPyro objects through intersphinx. For NumPyro distributions, use `{py:class}` for classes and `{py:func}` for factories documented as functions. Ordinary links are appropriate for the distribution catalogue.
- Follow `AGENTS.md` for Google-style docstrings: single backticks cross-reference public names; double backticks mark literal code or objects outside the API reference. Every public name in a signature or docstring must be exported and documented in `docs/api/`; do not hide missing targets with `nitpick_ignore`.

## Maintenance checks

Follow [the build and full-rebuild instructions](../docs/README.md) to check changes. Warnings are errors. A fresh Sphinx build checks all references; force code execution when the library or example parameters change, because cached cells may otherwise retain old outputs.

Add published pages to the appropriate toctree in `docs/index.md`. Keep unreviewed drafts explicitly excluded in `docs/conf.py`; a page outside both the navigation and exclusions fails the build. Development plans belong in `design/`.

Branding assets live in `docs/_static/branding/`. Preserve the Flush mark, Jost wordmark and Cynosural palette: midnight blue `#0A1F3D`, pale blue `#A9C6E2` and white. SVGs are the display sources, with lettering converted to paths.

## Future work

- **Tutorials — low priority.** Revisit the existing drafts in `docs/tutorials/`, choose the useful problems and their order, and review or rewrite them before publication. The newsvendor draft overlaps the quickstart. When tutorials are ready, update `exclude_patterns`, the home-page card and toctree, and the quickstart and README links together. Tutorial issues and possible extensions are recorded in [TODO.md](../TODO.md). The oil field draft now models `no_result` explicitly when no test is run and explains the warnings and default actions for impossible combinations; editorial and source review remains pending.
- **Further reading — optional, later.** Keep the candidate references below for a possible Resources page, grouped by topic with a short explanation of each source. Create the page when there is a useful selection to publish. Plain links and author–year citations are sufficient; verify bibliographic details before publication.

Future capabilities and related documentation, including learning from data, are tracked in [TODO.md](../TODO.md).

## References

Candidates for a future Resources page, grouped by topic. Verify details before publishing. Introductory material also includes the decision-analysis blog series ([I](https://ferjorosa.github.io/blog/2025/06/08/decision-theory-I.html), [II](https://ferjorosa.github.io/blog/2025/07/04/decision-theory-II.html), [III](https://ferjorosa.github.io/blog/2025/08/07/decision-theory-III.html)), [NumPyro documentation](https://num.pyro.ai/en/stable/) and [pyAgrum documentation](https://pyagrum.readthedocs.io).

- **Influence diagrams and their evaluation**
  - Howard, R. A. and Matheson, J. E. (1984). Influence diagrams.
  - Shachter, R. D. (1986). Evaluating influence diagrams. Operations Research.
  - Jensen, F., Jensen, F. V. and Dittmer, S. L. (1994). From influence diagrams to junction trees. UAI.
- **LIMIDs**
  - Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving decision problems with limited information. Management Science.
  - Nilsson, D. and Lauritzen, S. L. (2000). Evaluating influence diagrams using LIMIDs. UAI.
- **Continuous and mixed influence diagrams**
  - Shachter, R. D. and Kenley, C. R. (1989). Gaussian influence diagrams. Management Science.
  - Lauritzen, S. L. (1992). Propagation of probabilities, means and variances in mixed graphical association models. JASA.
  - Madsen, A. L. and Jensen, F. (2005). Solving linear-quadratic conditional Gaussian influence diagrams. IJAR.
  - Cobb, B. R. and Shenoy, P. P. (2008). Decision making with hybrid influence diagrams using mixtures of truncated exponentials. EJOR.
  - Li, Y. and Shenoy, P. P. (2010). Solving hybrid influence diagrams with deterministic variables. UAI.
- **Monte Carlo decision analysis**
  - Bielza, C., Müller, P. and Ríos Insua, D. (1999). Decision analysis by augmented probability simulation. Management Science 45(7), 995–1007.
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
