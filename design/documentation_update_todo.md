# Documentation update checklist

Update the existing documentation for the policy-gradient-only API, preserving the ice cream shop story, the author's writing style, and the progression between pages. Work through this checklist step by step and record checks as they are completed. [Documentation maintenance](documentation_plan.md) remains the source for site structure and writing conventions; this file tracks the migration work.

## Current status

| Step | Status |
| --- | --- |
| Starting-point inventory and baseline build | Complete. The baseline build reported 83 warnings. |
| API reference, public type targets, and navigation | Complete. The updated API pages produce no warnings, and their checked local links resolve. |
| Quickstart | Complete. Both models use a declared family, display fitted action probabilities in text and grouped bar charts, and execute successfully. |
| Building a diagram | Complete. Uses `TabularSoftmax`, retains the construction and editing examples, and executes successfully. |
| Distributions and utilities | Next. Update decision configuration and the comparison of combined and split utilities. |
| Remaining guides, resources, README, and figures | Pending. Follow the page-by-page checklist below. |
| Full strict documentation build | Failing with 13 warnings: five notebook failures with paired traceback warnings, plus three old references in the Solving guide. |

The API reference, quickstart, and Building a diagram guide are complete, but the documentation migration is still in progress. The current Building a diagram update is uncommitted. The build evidence and checks are recorded in the [verification log](#verification-log).

## Starting point

- [x] Create `docs/policy-gradient-update` from the latest `codex/policy-gradient-engine` tip after #69 was merged into #68 (`0b7f054`).
- [x] Create this checklist without changing published pages.
- [x] Inventory outdated API names, solver descriptions, imports, examples, outputs, and figure scripts across the published site and README. Findings are recorded under [Baseline inventory](#baseline-inventory).
- [x] Run the current strict documentation build and record the failures as the baseline. Results are recorded under [Baseline build](#baseline-build).

The starting-point audit is complete. Its findings are retained as the baseline; the completed migrations are listed in the current status above. The remaining items stay open until their content and examples have been checked.

## Preserve throughout

- Keep the existing narrative and explanations where they still apply. Change the behavior, terminology, code, or interpretation that the new API requires; avoid a wholesale rewrite.
- Preserve the shared uncertain-demand model and its Bayesian prior, the hotel-booking extension, and the promotion decision in the multiple-decisions guide. Keep intentional differences between pages explicit.
- Keep the quickstart and README order menu at 0–200. Preserve the Solving guide's expansion to 400 and its lesson about checking boundary answers, while checking whether the fitted stochastic results still support the explanation.
- Introduce policy families, stochastic rules, and training settings beside the decisions where readers need them. Explain types through their role in the example rather than presenting a catalogue in the walkthrough.
- Use the existing voice: complete sentences, plain words, “we” for the reader and author, and “the owner” or “the shop” for the example's decisions. Let Markdown lines flow.
- Derive every quoted numerical result from executed code. Keep code, prose, displayed outputs, and figures consistent.

## API changes across the documentation

The API reference now covers the current contract. Unchecked items in this section track the work still needed in walkthroughs, resources, README examples, and figures; they do not mean those topics are missing from the reference.

- [ ] Replace solver selection and the removed backward-induction and scan APIs with the single `solve()` workflow. Remove references to `method`, `SolveMethod`, `SolverName`, and `Solution.method` where they describe current behavior.
- [ ] Declare a `PolicyFamily` on each decision used by `solve()`. Explain when to use `TabularSoftmax` or `LinearSoftmax`, including their information requirements.
- [ ] Replace `SoftmaxPolicy` and `TabularPolicy` with `CategoricalPolicy`. Explain the distinction between a trainable family, a fitted rule, and the joint policy.
- [ ] Update `Solution.render()` calls and explanations: fitted rules retain the labels, so rendering no longer takes a diagram.
- [ ] Explain `num_samples`, `num_steps`, `learning_rate`, and `rng_key` using the current signatures and defaults. Both public entry points default to 10,000 samples; in `solve()`, the budget applies to each training update and to independent final evaluation.
- [x] Document `ActionTable`, `DecisionRule`, `Policy`, `FittedPolicy`, `DecisionInformation`, and `PolicyParameters` with their intended meanings. Distinguish the policies accepted by `infer()` from those returned by `solve()`, and single information assignments from batches.
- [ ] Update errors and examples for `InferenceError`, `SolveError`, and configuration `ValueError`s according to the implementation.
- [ ] Repair imports and cross-references for the graph, diagram, node, and policy packages. Keep internal helpers out of the public API reference.
- [ ] Explain that fitted actions remain stochastic, expected utility is estimated on independent scenarios, and training gives no convergence or optimality guarantee. Distinguish evaluation noise from policy quality and explain poorly visited or impossible tabular rows where relevant.
- [ ] Preserve the current inference and distribution boundaries: finite categorical actions, JAX-traceable functions, explicit information and memory arcs, count-support limits, and unsupported posterior inference through relevant function or fitted policy rules.

## Page-by-page work

For each page, read the existing prose first, update the example and its interpretation together, execute its code, and check references and any associated figures. Mark a page complete only after those checks pass. Keep the existing site order unless a concrete change is agreed during the work.

- [x] `docs/api/`: update the entry points, results, errors, policy families, rules, aliases, and graph references. Replace or retire the obsolete individual-solvers reference and repair its navigation and incoming links. Ensure public names used in signatures and docstrings have resolvable targets.
- [x] `docs/getting_started/quickstart.md`: adapt the small ice cream example without expanding it into an optimizer tutorial; update the policy display and result interpretation.
- [x] `docs/user_guide/building_a_diagram.md`: introduce the required decision configuration while preserving the build, validate, and draw progression.
- [ ] `docs/user_guide/distributions_and_utilities.md`: check model consistency, JAX requirements, inference claims, examples, and demand figures.
- [ ] `docs/user_guide/solving.md`: explain fitting a policy, choosing a family, controlling training and evaluation, reading stochastic results, and checking the expanded order menu.
- [ ] `docs/user_guide/inference.md`: evaluate fitted policies with the current API while preserving the intentional booking-information variant, aligned draws, comparisons, and inference figures.
- [ ] `docs/user_guide/multiple_decisions.md`: preserve the promotion story and explain joint fitting with explicit information parents and memory.
- [ ] `docs/resources/common_pitfalls.md` and `docs/resources/count_distributions.md`: update standalone examples and diagnostics; remove obsolete solver advice while retaining relevant modeling and inference limits.
- [ ] `docs/index.md`, `docs/getting_started/installation.md`, and `README.md`: align claims, quick examples, links, and terminology with the updated guides. Keep the README example consistent with the quickstart.
- [ ] Review excluded tutorial drafts for stale references and record remaining work separately. Keep unreviewed tutorials excluded; publishing or rewriting them is outside this migration's initial scope.
- [ ] Finalize `design/documentation_plan.md` when the migration is complete. The new API structure and current migration status are already recorded there. Check `docs/README.md` if build instructions change.

## Completion checks

- [ ] Re-run figure scripts affected by example changes and visually inspect their outputs. Confirm their parameters match the corresponding pages.
- [ ] Search published sources, docstrings, and figure scripts for removed names, solver-selection claims, outdated defaults, and deterministic-policy assumptions. Check remaining matches in context.
- [ ] Build with warnings treated as errors and force every published example to execute, following [the build instructions](../docs/README.md):

  ```bash
  uv run sphinx-build -W -E -a -D nb_execution_mode=force docs docs/_build/html
  ```

- [ ] Resolve broken references, missing public type targets, execution errors, and navigation issues rather than suppressing warnings or disabling execution.
- [ ] Review the rendered pages for prose, output readability, policy tables, figures, links, and continuity of the shared story. Re-check numerical claims against the executed outputs.
- [ ] Run Ruff lint, formatting, type checks, and the test suite if example scripts or library docstrings were changed; run `git diff --check` for all changes.
- [ ] Record the final build result, relevant test results, and any explicitly deferred work below.

## Verification log

### Baseline inventory

Reviewed on 2026-10-26 against `0b7f054`, including published Markdown and API sources, the README, the four figure scripts, and searches through the excluded tutorial drafts. API claims were checked against the current entry points, public exports, policy classes, and policy binding. Findings below are source observations unless the build results explicitly confirm execution.

This table records the original state before migration. Completed work is recorded in the update entries below; the remaining walkthrough findings stay pending.

| Area | Findings and migration work |
| --- | --- |
| `docs/api/pylimid.rst` | Still lists `SoftmaxPolicy`, `TabularPolicy`, `SolverName`, and `SolveMethod`. Its `Policy` definition omits fixed actions and describes the old result types. Add reference coverage for `CategoricalPolicy`, `PolicyFamily`, `SolveError`, and all six public policy aliases. |
| `docs/api/solvers.rst`, `docs/api/index.rst`, and `docs/index.md` | The individual-solvers page, its description, and its navigation still advertise removed solver and solvability functions. Decide its replacement or retirement when updating the API section, then repair incoming links. |
| `docs/api/graph.rst` | Public graph re-exports still resolve, but the prose describes snapshots for individual solvers. Check links against the new owning packages and retain user-facing graph APIs. |
| `README.md` and `docs/getting_started/quickstart.md` | Decisions lack `policy_family`. Policy-reading loops call `.items()` on what is now a fitted rule, and prose promises particular best actions. Preserve the quickstart's initial model, its later Bayesian-prior extension, and the README's prior-based version; re-run their results and explain stochastic policies. |
| `docs/user_guide/building_a_diagram.md` | Examples execute, but the decision remains without a family and prose still presents table solvers and `method="policy_gradient"`. Keep incomplete model construction distinct from the configuration needed for fitting. Validation examples do not currently check for a missing policy family; `solve()` does. |
| `docs/user_guide/distributions_and_utilities.md` | The solve example lacks a family and compares fitted policies with dictionary equality. Replace this with an appropriate comparison of probabilities and utility estimates. Preserve the explanation of additive utilities and the JAX guidance. |
| `docs/user_guide/solving.md` | Contains deterministic `.items()` / `.values()` policy access, `render(diagram)`, solver selection, solvability checks, backward-induction warnings, and removed policy names. Existing training examples still pass `method=` and describe custom families as future work despite the public `PolicyFamily` interface. The errors section has obsolete exception claims. Preserve the order-menu lesson and review the separate cart and signal examples for their teaching value during the page update. |
| `docs/user_guide/inference.md` | The initial solve lacks a family; both policy displays pass a diagram to `render()`. The sunny/high-bookings profit example also passes a fitted policy together with `observed=`, which the current binder explicitly rejects. Preserve that question through a supported scenario representation when revising the page; choosing deterministic argmax actions would change the policy. Re-check all profit figures, spikes, and numerical claims. The later demand-only query prunes the decision and has a different support boundary. |
| `docs/user_guide/multiple_decisions.md` | Both decisions lack families. The explanation treats rules as deterministic tables, assumes the promotion always runs, and explains memory through the removed `is_solvable` / automatic scan choice. Preserve the promotion story and explicit-memory lesson using joint stochastic fitting. |
| `docs/resources/count_distributions.md` | The capacity decision lacks a family and uses `render(arrivals)`. The claim that policies cannot use an observed numerical count is too broad: `TabularSoftmax` requires categorical inputs, whereas `LinearSoftmax` accepts numerical information. Keep these policy-family limits separate from count-posterior limits. |
| `docs/resources/common_pitfalls.md` | Both examples execute, and `probe_discrete_parents()` remains a supported diagram method. Retain the indexing and parent-effect explanations; no API break was found in these examples. |
| `docs/index.md`, `README.md`, and installation/build instructions | Homepage and README accuracy claims need to distinguish evaluation noise from optimization quality. Installation and `docs/README.md` have no engine-specific break found in this audit; retain the existing CUDA and Graphviz guidance and strict-build workflow. |

Current public defaults are `num_samples=10_000` for `infer()` and `solve()`, with `num_steps=500` and `learning_rate=0.05` for `solve()`. Many guides explicitly request 50,000 samples; these are chosen budgets rather than stale defaults. Their cost and purpose need review because solving now uses that many scenarios at every update as well as in final evaluation. Changing sample counts alone cannot establish optimizer convergence or policy quality.

The published examples mostly use supported package-root imports. The removed `from pylimid.solver import is_solvable` imports in the Solving and Multiple decisions guides need replacement; no old graph-node or public NumPyro-integration module imports were found in the published examples.

#### Figures

- `docs/_static/figures/inference/plot_recommended_policy_profit.py` executes the inference guide's code cells directly. It inherits the missing-family failure and the later unsupported fitted-policy/evidence combination. Its four profit, comparison, and demand images need regeneration after the guide is corrected.
- `docs/_static/figures/quickstart/plot_demand.py`, `docs/_static/figures/quickstart/plot_busyness_prior.py`, and `docs/_static/figures/distributions_and_utilities/plot_demand_comparison.py` plot distributions directly and do not call the removed solver APIs. Their parameters match the relevant page definitions in this source review. They were not rerun or visually checked during the starting-point audit.

#### Excluded tutorial backlog

`docs/conf.py` excludes the tutorial directory, so these drafts are outside the baseline build and remain unpublished. Search findings are recorded here without marking their later review complete:

- `oil_field.md`: automatic backward induction, impossible-row warnings, `solution.method`, and deterministic recommendations.
- `newsvendor.md`: backward-induction claims, `solution.method`, deterministic table iteration, and an explicit 300,000-sample solve budget to revisit.
- `logistics_center.md`: backward-induction claims, `solution.method`, deterministic rule indexing, and an explicit 400,000-sample solve budget to revisit.
- `bayesian_networks.md`: inference examples need checking against the current API and defaults; prose incorrectly describes 2,000 draws as the engine's default output.

### Baseline build

Executed on 2026-10-26 with Sphinx 8.2.3, fresh Sphinx processing, and forced notebook execution:

```bash
UV_CACHE_DIR=/private/tmp/pylimid-uv-cache uv run --offline sphinx-build -W -E -a -D nb_execution_mode=force docs docs/_build/html
```

Result: exit code **1**, with **83 Sphinx warnings treated as errors**. HTML was produced, but this is a failed build. Python, JAX, and NumPyro intersphinx inventories loaded successfully. The full local log is `docs/_build/documentation-baseline.log`; notebook tracebacks are under `docs/_build/html/reports/`. These generated files are ignored by Git.

| Diagnostics | Observed cause |
| --- | --- |
| 24 import warnings | Eight removed names are reported across autosummary generation, autodoc, and source summaries: `SoftmaxPolicy`, `TabularPolicy`, `backward_induction_solve`, `batched_solve`, `scan_solve`, `policy_gradient_solve`, `is_solvable`, and `solvability_order`. |
| 12 notebook warnings | Six pages fail execution, each producing a failure warning and a saved-traceback warning. First failures are listed below. |
| 8 navigation warnings | Existing ignored autosummary pages for removed names are no longer included in a toctree. Regenerate or remove obsolete generated stubs when updating the API reference. |
| 39 reference warnings | Old names still appear in prose and new public objects have no reference targets, including `PolicyFamily`, `CategoricalPolicy`, `DecisionInformation`, `PolicyParameters`, `FittedPolicy`, and `SolveError`. |

These are diagnostic counts, including repeated reports and existing generated stubs, rather than 83 independent content defects. Remaining aliases also need explicit API documentation even where the build has not yet reported a missing target.

| Notebook | First execution failure |
| --- | --- |
| `docs/getting_started/quickstart.md` | `ValueError`: decision `order` must declare a policy family. |
| `docs/resources/count_distributions.md` | `ValueError`: decision `capacity` must declare a policy family. |
| `docs/user_guide/distributions_and_utilities.md` | `ValueError`: decision `order` must declare a policy family. |
| `docs/user_guide/inference.md` | `ValueError`: decision `order` must declare a policy family. |
| `docs/user_guide/multiple_decisions.md` | `ValueError`: decision `promotion` must declare a policy family. |
| `docs/user_guide/solving.md` | `ValueError`: decision `order` must declare a policy family. |

`docs/resources/common_pitfalls.md` and `docs/user_guide/building_a_diagram.md` execute successfully. Execution stops at the first failing cell in each other notebook, so later rendering calls, policy access, imports, and evidence combinations have not been validated by this build. README snippets, excluded tutorials, and figure scripts are outside Sphinx's notebook execution and were reviewed as source only.

The initial restricted attempt exited with code 2 because Jupyter could not bind a local kernel socket. The baseline above was rerun with the needed access and reached the actual documentation failures. The initial log is retained as `docs/_build/documentation-baseline-restricted.log`.

At the baseline stage, only this checklist was edited; published sources and library code were unchanged. No commit was made.

### API reference update

Completed on 2026-10-26 without committing. `docs/api/pylimid.rst` now covers the current entry points, sampling and training settings, results, aligned posterior draws, and errors. `docs/api/policy.rst` replaces the removed `docs/api/solvers.rst`, documenting policy families, fitted rules, and the six public policy aliases. Graph reference pages use the owning public `graph.nodes` and `graph.diagram` packages. The API index, site navigation, and documentation maintenance guide reflect this structure.

The aliases have explicit public reference targets and canonical defining-module paths. Class and function documentation remains generated from the library's docstrings. No library code, docstrings, or walkthrough examples were changed. The earlier generated API stubs were moved to a temporary backup and regenerated so removed names do not persist as source pages.

- Re-ran the full strict build with fresh Sphinx processing and forced notebook execution, using the same command as the baseline. Result: exit code **1**, with **15 warnings**, down from 83. Log: `docs/_build/documentation-api-update.log`.
- The API reference produces **no warnings**. The remaining diagnostics are the six unchanged notebook failures, each with its paired traceback warning, and three removed-name references in `docs/user_guide/solving.md`: `is_solvable`, `SoftmaxPolicy`, and `TabularPolicy`. Those belong to the pending walkthrough migration.
- Confirmed that every public export from `pylimid`, `pylimid.graph`, and `pylimid.policy` has a reference target, except the version constant. Checked the fitted-rule methods and the `PolicyFamily` extension methods explicitly.
- Checked **2,837 local page and anchor links across 24 rendered API pages**, including generated pages; all resolve. Inspected the rendered type signatures and main reference text.
- Ruff lint, repository-wide formatting, type checks, and `git diff --check` pass. No runtime tests were rerun for these documentation-only changes.

The broader API-change checklist above remains open where it also covers walkthroughs, prose, figures, and examples outside the reference section. The complete site still fails its strict build until those pages are migrated.

### Quickstart update

Completed on 2026-10-26 without committing the changes. The initial ice cream model, its 0–200 order menu, and the Bayesian-prior extension are preserved. The `order` decision now declares `TabularSoftmax()`, and both solutions use `Solution.render()` to show their fitted action probabilities. The prose explains the stochastic rule, independent utility evaluation, default training budget, and limits of approximation without adding a separate optimizer tutorial.

- Executed all **nine code cells**, including both solves, directly and through Sphinx. Both diagrams validate with no problems.
- With the default settings, the initial model estimates expected profit at **498.09**. Its most likely orders are 60 on cloudy days with probability **0.979**, and 200 on sunny days with probability **0.935**. The prior-based model estimates **600.56**, favouring 80 with probability **0.994** and 200 with probability **0.991**, respectively. The page interprets these probabilities without converting the fitted policy to deterministic actions.
- Replaced the former explicit 50,000-sample calls with the public defaults: 10,000 scenarios per update, 500 updates, and 10,000 independent evaluation scenarios. The page distinguishes training quality from evaluation noise.
- Re-ran the strict full build with fresh Sphinx processing and forced notebook execution. Result: exit code **1**, with **13 warnings**, down from 15. The quickstart executes successfully and produces no warnings. Log: `docs/_build/documentation-quickstart-update.log`.
- The remaining failures are in Count distributions, Distributions and utilities, Inference, Multiple decisions, and Solving, each with a paired traceback warning. The three removed-name references in Solving remain pending.
- Verified that the executed notebook contains the same nine code cells as the source, with no error outputs, and that its displayed probabilities and expected profits support the prose. Checked **117 local page and anchor links** and **six image references** in the rendered page; all resolve.
- Visually inspected the existing demand and busyness-prior plots and checked their script parameters against the page. The distributions and plot parameters are unchanged, so the figures and their scripts were retained.
- Ruff lint, repository-wide formatting, type checks, and `git diff --check` pass. Library code and figure scripts are unchanged; the runtime verification for this step is the execution of the quickstart examples.

### Quickstart policy figures

Added on 2026-10-26 without committing. The Solving section and Bayesian-prior extension now each show a grouped bar chart of the fitted order probabilities, using blue-grey for cloudy days and orange for sunny days to match the existing demand figure. Each colour represents a separate conditional distribution summing to 100%; bar heights come directly from `CategoricalPolicy.probabilities()`, with no histogram sampling or deterministic conversion.

- Added `docs/_static/figures/quickstart/plot_policy.py`. It executes the quickstart's code cells to obtain both fitted policies, keeping the model parameters and fitting settings tied to the page.
- Generated and visually inspected `order_policy.png` and `prior_order_policy.png`. The charts show the cloudy-day shift from 60 to 80 and the sunny-day concentration at 200, with the same probabilities reported in the text and rendered policy output.
- Re-ran the strict full build with forced notebook execution. The quickstart succeeds with no warnings, and both new figures appear in the rendered page, which now has eight image references. The full build still exits with code **1** and the same **13 warnings** from pending pages. Log: `docs/_build/quickstart-policy-build.log`.
- Full test suite: **404 passed**. Ruff lint, formatting, type checks, and `git diff --check` pass. Figure-generation and test logs are under `docs/_build/quickstart-policy-figures.log` and `docs/_build/quickstart-policy-tests.log`.

### Building a diagram update

Completed on 2026-10-26 without committing. The example declares `TabularSoftmax()` as in the quickstart. Removed the old solver-selection discussion from the information-set note and added one sentence distinguishing diagram validation from the family requirement checked by `solve()`. Construction, editing, rendering, and the hotel-booking extension are preserved.

- All **12 code cells** execute successfully. Verified the missing-distribution and stale-signature messages, both empty validation results after repairs, and generated diagrams.
- The page produces no warnings. Checked **117 local page and anchor links** and **four image references**; all resolve.
- The strict full build still exits with code **1** and the same **13 warnings** from pending pages. Log: `docs/_build/building-diagram-update.log`.
- Ruff lint, formatting, type checks, and `git diff --check` pass. No library code or figure scripts changed.

## Next step

Update `docs/user_guide/distributions_and_utilities.md`: declare the policy family and replace fitted-policy dictionary equality with a suitable comparison of action probabilities and expected utility. Preserve the distribution explanations, shared model, additive-utility example, and JAX guidance.
