# Documentation maintenance

Maintenance guide for the published documentation in `docs/` and the project `README.md`. The engine uses policy-gradient-only optimization. The API reference, quickstart, four core user guides, resource pages, site overview, and README describe the current contract. The quickstart and guides use the mixed categorical/continuous ice cream shop example. Read this before changing published pages, and update it when the structure or conventions change. Public API details belong in docstrings and the API reference. Build instructions live in [docs/README.md](../docs/README.md).

## Site structure and scope

The published order is **Getting started → User guide → Resources → API reference**. Tutorials will sit between User guide and Resources when ready.

- **Getting started** covers installation and a quickstart: one small problem through build, validate, draw and solve.
- **User guide** continues the quickstart across Building a diagram, Distributions and utilities, Solving a diagram, and Inference. Joint decisions and explicit memory are introduced within this sequence, so there is no separate Multiple decisions page.
- **Tutorials** tell the story of a complete alternative problem, with figures and interpretation of the results.
- **Resources** contains Common pitfalls, Count distributions and References, directly under the sidebar heading. Example pages have their own context and imports. Count distributions collects count-specific examples and limitations. References is a reading guide covering introductions, influence diagrams and limited memory, continuous and mixed variables, simulation and policy gradients, probabilistic programming, and software. Each section introduces the topic before listing verified citations with links and short explanations of their relevance.
- **API reference** covers inference and solving, building diagrams, and policy families and rules. Classes, functions and methods are generated from docstrings; public type aliases are explained alongside their owning APIs. The policy page replaces the individual-solvers page.

Contributor documentation belongs in `CONTRIBUTING.md`, `AGENTS.md` and `design/`. Explain architecture on the published site only when it helps users understand execution or trust a result. Link to NumPyro for distributions and MCMC; focus PyLIMID explanations on information sets, explicit memory, policy families and fitting, callable parents and decision-modelling patterns. Introduce theory beside the behaviour it explains, with a citation rather than a textbook derivation. The current optimizer objective and verification are described in the [solver design](solver/algorithms.md).

### Shared example

The README, quickstart and user guide follow the ice cream shop. The owner chooses a categorical promotion decision the evening before, then orders a continuous quantity of ice cream after seeing the morning weather and remembering the promotion choice. Quantities are measured in kilograms, with a freezer capacity of 40 kilograms. The quickstart adds uncertain seasonal busyness, and the guides extend that model with unobserved hotel bookings. The Count distributions page uses standalone examples so it does not interrupt this story. Keep the shared model consistent across pages while preserving intentional variants, such as revealing bookings before ordering in the inference guide.

The quickstart and initial guide model use a bounded TruncatedNormal order policy with learned location and fixed scale 0.25 kilograms. The Solving guide compares that fit with scale 1 kilogram and retains the wider family for Inference. The order domain and truncation bounds stay at 0–40 kilograms throughout. Model parameters and numerical outputs live in the pages and their figure scripts.

### Guide progression

Introduce the owner's problem before terminology, explain each step as we reach it, and connect the results to the shop's choices. Keep fitting details brief in the quickstart and develop them in the later guides.

The progression is quickstart, Building a diagram, Distributions and utilities, Solving a diagram, and Inference. Building a diagram, at `docs/user_guide/building_a_diagram.md`, covers construction, validation, timing and explicit memory. It restores the promotion-to-order memory arc after demonstrating its removal, then adds unobserved hotel bookings.

Distributions and utilities, at `docs/user_guide/distributions_and_utilities.md`, continues that model. It distinguishes categorical action indices from continuous quantities, compares Gamma and nonnegative Gaussian demand in kilograms, and separates demand uncertainty from the order policy. It replaces the combined `profit` utility with `revenue`, `stock_cost` and `promotion_cost`, demonstrating their equality on one day without introducing another fitting exercise. The final model retains both decisions, the promotion-to-order memory arc and unobserved hotel bookings.

Solving, at `docs/user_guide/solving.md`, starts with that exact model and jointly fits the two decision policies using the sum of its three utilities. It interprets categorical probabilities and continuous order distributions, compares training budgets and random keys, and demonstrates a different fixed order scale. The initial narrow-scale fit rarely promotes. The scale-1 fit finds a policy that usually promotes and has higher estimated profit. The page keeps this new family and `solution` for subsequent examples. Its final order domain is still 0–40 kilograms, with fixed truncation bounds 0 and 40, learned location and fixed scale 1.0. Information parents and utility nodes are unchanged.

Inference, at `docs/user_guide/inference.md`, continues with the scale-1 family and sums aligned draws from the three utility nodes. It distinguishes outcome variation from sampling error, conditions on sunny weather and high bookings, and separately refits a policy that can use a morning hotel report. It restores the original information parents after the report example. It compares a fixed 16-kilogram order while retaining the fitted promotion rule, evaluates demand with a known promotion choice, and updates beliefs about bookings and busyness from observed demand. Chance evidence conditions the model without granting extra decision information or retraining policies.

## Writing conventions

### Voice and terminology

- Write complete sentences for readers new to the library. Introduce a concrete problem before code and explain terms at first use. Prefer plain words and active voice where the actor matters.
- Use “we” for the author and reader building models and interpreting results. Use “the owner” or “the shop” for the example's decisions, avoiding pronouns for the owner. Introduce scenarios directly, for example “Suppose today is sunny”.
- Use present tense for walkthrough steps and API behaviour; past tense for earlier steps and conditionals for hypothetical alternatives.
- Keep pages short by choosing what to include. Keep explanations, code and output that help readers understand the example; put reusable debugging examples in Common pitfalls.
- Follow the revised quickstart's pacing: explain what the current step needs, keep captions brief, and leave forward navigation to the end of the page.
- Do not use semicolons. Let related ideas flow through natural conjunctions, or use separate sentences when the thought changes.
- Preserve character motivations and the explanations connecting choices to outcomes when updating APIs. Remove obsolete solver details without flattening the story around them.
- Write **PyLIMID** in prose and `pylimid` for the package, imports, extras and URLs. Use **node** for diagram structure and **variable** for the represented value. A diagram can combine discrete and continuous variables; a single variable is not “mixed”.
- Format specific node names, Python identifiers and literal state labels as code; keep ordinary concepts in plain text.
- Let Markdown lines flow; do not wrap at 80 characters.

### Accuracy and examples

- Check API claims against the implementation and API reference. `solve` always jointly trains the declared policy families. `LinearSoftmax` and `TabularSoftmax` fit categorical decisions, and `DistributionPolicy` composes scalar continuous distributions with parameter models such as `Linear`. Custom families implement `PolicyFamily`. There is no solver selector. Each rule receives only its declared information parents, including explicit memory arcs. Fitted policies contribute learned action probabilities or densities to forward evaluation and posterior conditioning. Action domains do not automatically truncate or scale distributions. Describe independent utility estimates as approximate results, with no convergence or optimality certificate. Supplied deterministic function policies still support forward evaluation only. Keep chance distribution contracts in `ChanceNode`.
- Distributions and utilities must be JAX-traceable. Do not describe them as arbitrary Python code; explain the restrictions on casts and Python control flow where relevant.
- Both `infer` and `solve` return Monte Carlo estimates. Distinguish exact discrete enumeration as a method from the accuracy of the returned summaries, including in mixed models. Explain sampling reliability where readers use estimates. Explain that rarely visited policy-table rows can receive little training and impossible rows do not affect expected utility.
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

Add published pages to the appropriate toctree in `docs/index.md`. Keep unreviewed drafts explicitly excluded in `docs/conf.py`; a page outside both the navigation and exclusions fails the build. Contributor design notes belong in `design/`.

Branding assets live in `docs/_static/branding/`. Preserve the Flush mark, Jost wordmark and Cynosural palette: midnight blue `#0A1F3D`, pale blue `#A9C6E2` and white. SVGs are the display sources, with lettering converted to paths.

### Latest verification

Verified on 2026-10-28. The full test suite passes with 534 tests, including the regression for joint conditioning with categorical and bounded continuous policies. Ruff lint, formatting, type checks and `git diff --check` pass. The strict Sphinx build with fresh processing and forced execution of all published examples succeeds without Sphinx warnings. The mixed-decision quickstart and four guides replace the original introductory pages at their existing paths. Their figure scripts execute from the canonical directories, and the README example reproduces its stated profit. All rendered local links, heading anchors and images resolve.

## Future work

- **Tutorials — low priority.** Revisit the existing drafts in `docs/tutorials/`, choose the useful problems and their order, and review or rewrite them before publication. The newsvendor draft overlaps the quickstart. Evaluate continuous-consequence variants of the logistics tutorial and its optional report-probability calibration during that review. Migrate the decision tutorials to fitted stochastic policies and revisit the large training sample budgets in the newsvendor and logistics drafts. Check the Bayesian-networks draft against the current inference API and defaults. When tutorials are ready, update `exclude_patterns`, the home-page card and toctree, and the quickstart and README links together. Editorial and source review remains pending.
- **Further reading — optional, later.** Extend the [References page](../docs/resources/references.md) when another source helps explain a modelling or analysis question. Keep the topic introductions and short annotations rather than growing an unstructured bibliography. Verify authors, publication year, title and venue against publisher or author records, and cite the edition or version linked. Explain historical solution methods without suggesting that PyLIMID implements every algorithm listed.

Remaining reliability and documentation work is tracked in [TODO.md](../TODO.md).

## Additional reference candidates

The verified bibliography lives on the [References page](../docs/resources/references.md). It cites Howard and Matheson's 2005 journal publication and the second editions of the introductory books where applicable. The following sources remain candidates for a more specialised discussion. Verify their bibliographic details before publication.

- Nilsson, D. and Lauritzen, S. L. (2000). Evaluating influence diagrams using LIMIDs. UAI.
- Lauritzen, S. L. (1992). Propagation of probabilities, means and variances in mixed graphical association models. JASA.
- Madsen, A. L. and Jensen, F. (2005). Solving linear-quadratic conditional Gaussian influence diagrams. IJAR.
- Kearns, M., Mansour, Y. and Ng, A. (1999). A sparse sampling algorithm for near-optimal planning in large Markov decision processes. IJCAI.
- Bellman, R. (1957). Dynamic Programming.
