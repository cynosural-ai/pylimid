# Documentation update checklist

Update the existing documentation for the policy-gradient-only API, preserving the ice cream shop story, the author's writing style, and the progression between pages. Work through this checklist step by step and record checks as they are completed. [Documentation maintenance](documentation_plan.md) remains the source for site structure and writing conventions; this file tracks the migration work.

## Current status

| Step | Status |
| --- | --- |
| Starting-point inventory and baseline build | Complete. The baseline build reported 83 warnings. |
| API reference, public type targets, and navigation | Complete. The updated API pages produce no warnings, and their checked local links resolve. |
| Quickstart | Complete. Both models use a declared family, display fitted action probabilities in text and grouped bar charts, and execute successfully. |
| Building a diagram | Complete. Uses `TabularSoftmax`, retains the construction and editing examples, and executes successfully. |
| Distributions and utilities | Complete. The combined and split utility models produce matching fitted probabilities and expected profit. |
| Solving a diagram | Complete. Fitted-policy summaries, training comparisons, wider order menu, cart coordination, and continuous information examples execute successfully. |
| Inference | Complete. Uses fitted policies with direct chance-node conditioning, and the examples, figures, and API-reference boundaries have been checked. |
| Multiple decisions | Complete. Jointly fits the promotion and order rules, explains explicit memory and rarely used rows, and evaluates aligned total-profit draws. |
| Common pitfalls | Complete. Retains indexing and parent-effect checks, and adds state-index, joint-draw, and fitted-policy examples. |
| Count distributions | Complete. Fits capacity policies before and after observing a numerical count, and retains the supported posterior examples and limits. |
| Remaining site and README checks | Complete, awaiting review of the final changes. The README uses fitted rules, the overview pages distinguish training quality from estimation noise, and installation guidance has been verified. |
| Full strict documentation build | Passing with no Sphinx warnings and forced execution of all published examples. |

The published documentation migration is complete. The final README, overview, installation wording, and verification-record changes are uncommitted for review. Excluded tutorial drafts remain a separate migration task. The build evidence and checks are recorded in the [verification log](#verification-log).

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

The API reference, walkthroughs, resources, README example, and figure scripts now cover the current contract. The completed items below apply to published pages. Excluded tutorial work is recorded separately.

- [x] Replace solver selection and the removed backward-induction and scan APIs with the single `solve()` workflow. Remove references to `method`, `SolveMethod`, `SolverName`, and `Solution.method` where they describe current behavior.
- [x] Declare a `PolicyFamily` on each decision used by `solve()`. Explain when to use `TabularSoftmax` or `LinearSoftmax`, including their information requirements.
- [x] Replace `SoftmaxPolicy` and `TabularPolicy` with `CategoricalPolicy`. Explain the distinction between a trainable family, a fitted rule, and the joint policy.
- [x] Update `Solution.render()` calls and explanations: fitted rules retain the labels, so rendering no longer takes a diagram.
- [x] Explain `num_samples`, `num_steps`, `learning_rate`, and `rng_key` using the current signatures and defaults. Both public entry points default to 10,000 samples; in `solve()`, the budget applies to each training update and to independent final evaluation.
- [x] Document `ActionTable`, `DecisionRule`, `Policy`, `FittedPolicy`, `DecisionInformation`, and `PolicyParameters` with their intended meanings. Distinguish the policies accepted by `infer()` from those returned by `solve()`, and single information assignments from batches.
- [x] Update errors and examples for `InferenceError`, `SolveError`, and configuration `ValueError`s according to the implementation.
- [x] Repair imports and cross-references for the graph, diagram, node, and policy packages. Keep internal helpers out of the public API reference.
- [x] Explain that fitted actions remain stochastic, expected utility is estimated on independent scenarios, and training gives no convergence or optimality guarantee. Distinguish evaluation noise from policy quality and explain poorly visited or impossible tabular rows where relevant.
- [x] Preserve the current inference and distribution boundaries: categorical and scalar continuous actions with fixed support, JAX-traceable functions, explicit information and memory arcs, count-support limits, posterior conditioning through fitted policies, and forward-only evaluation of supplied deterministic function policies.
- [x] Document `Real`, `Interval`, `DistributionPolicy`, `ContinuousPolicy`, `ParameterModel`, and `Linear` in the API reference. The continuous-decision worked example and walkthrough remain deferred in the documentation maintenance plan.

## Page-by-page work

For each page, read the existing prose first, update the example and its interpretation together, execute its code, and check references and any associated figures. Mark a page complete only after those checks pass. Keep the existing site order unless a concrete change is agreed during the work.

- [x] `docs/api/`: update the entry points, results, errors, policy families, rules, aliases, and graph references. Replace or retire the obsolete individual-solvers reference and repair its navigation and incoming links. Ensure public names used in signatures and docstrings have resolvable targets.
- [x] `docs/getting_started/quickstart.md`: adapt the small ice cream example without expanding it into an optimizer tutorial; update the policy display and result interpretation.
- [x] `docs/user_guide/building_a_diagram.md`: introduce the required decision configuration while preserving the build, validate, and draw progression.
- [x] `docs/user_guide/distributions_and_utilities.md`: check model consistency, JAX requirements, inference claims, examples, and demand figures.
- [x] `docs/user_guide/solving.md`: explain fitting a policy, choosing a family, controlling training and evaluation, reading stochastic results, and checking the expanded order menu.
- [x] `docs/user_guide/inference.md`: evaluate fitted policies with the current API while preserving the intentional booking-information variant, aligned draws, comparisons, and inference figures.
- [x] `docs/user_guide/multiple_decisions.md`: preserve the promotion story and explain joint fitting with explicit information parents and memory.
- [x] `docs/resources/common_pitfalls.md`: retain the existing indexing and parent-effect diagnostics, and add standalone examples for state indices, joint draws, and changing fitted policies.
- [x] `docs/resources/count_distributions.md`: update standalone examples and diagnostics, and remove obsolete solver advice while retaining relevant modeling and inference limits.
- [x] `docs/index.md`, `docs/getting_started/installation.md`, and `README.md`: align claims, quick examples, links, and terminology with the updated guides. Keep the README example consistent with the quickstart.
- [x] Review excluded tutorial drafts for stale references and record remaining work separately. Keep unreviewed tutorials excluded; publishing or rewriting them is outside this migration's initial scope.
- [x] Finalize `design/documentation_plan.md` when the migration is complete. The new API structure and current migration status are already recorded there. Check `docs/README.md` if build instructions change.

## Completion checks

- [x] Re-run figure scripts affected by example changes and visually inspect their outputs. Confirm their parameters match the corresponding pages.
- [x] Search published sources, docstrings, and figure scripts for removed names, solver-selection claims, outdated defaults, and deterministic-policy assumptions. Check remaining matches in context.
- [x] Build with warnings treated as errors and force every published example to execute, following [the build instructions](../docs/README.md):

  ```bash
  uv run sphinx-build -W -E -a -D nb_execution_mode=force docs docs/_build/html
  ```

- [x] Resolve broken references, missing public type targets, execution errors, and navigation issues rather than suppressing warnings or disabling execution.
- [x] Review the rendered pages for prose, output readability, policy tables, figures, links, and continuity of the shared story. Re-check numerical claims against the executed outputs.
- [x] Run Ruff lint, formatting, type checks, and the test suite if example scripts or library docstrings were changed; run `git diff --check` for all changes.
- [x] Record the final build result, relevant test results, and any explicitly deferred work below.

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

### Distributions and utilities update

Completed on 2026-10-26 without committing. Added `TabularSoftmax()` to the decision. The comparison now displays the most likely stock levels and their maximum probabilities for cloudy and sunny days, using `jax.vmap` to evaluate both forecasts without a loop. Both calls use the default fitting budget. Distribution explanations, the additive-utility example, and JAX guidance are preserved.

- All **eight code cells** execute successfully. Both models show stock levels **80** and **200**, with maximum probabilities **0.9894665** and **0.9928226**, in cloudy/sunny order. Both report expected profit **618.7**.
- The page produces no warnings. The strict full build exits with code **1** and **11 warnings**, from the four remaining notebook failures and three old references in Solving. Latest log: `docs/_build/utility-recommendations-update.log`.
- Checked the rendered page's local links and image references. Visually inspected the existing demand-comparison figure; its parameters still match the page, so the figure and script are unchanged.
- Ruff lint, formatting, type checks, and `git diff --check` pass. No library code or figure scripts changed.

### Solving update

Completed on 2026-10-26 without committing. Updated the shared model, fitted-rule summaries, training settings, rare-row explanation, and errors for the single engine. Preserved the wider order-menu lesson and adapted the cart and continuous-signal examples. Removed the obsolete solver-choice discussion and repeated joint-signal, categorical-table, and mixed-family examples.

- All **eight code cells** execute successfully. After 100 updates, the most likely stocks remain 80 and 200, with probabilities about **55%** and **68%**; after 500 updates, both exceed **98%** in the two illustrated seeds.
- Expanding the menu to 400 produces a most likely sunny-day order of **260**, with probability **0.9257** and estimated expected profit **626.5**. The cart example favours Mia at the beach and Leo at the park; the continuous-signal example uses a fitted `CategoricalPolicy` with `LinearSoftmax`.
- The page produces no warnings. The strict full build now exits with code **1** and **six warnings**, all from the remaining Inference, Multiple decisions, and Count distributions notebook failures and their traceback reports. Removed-name references no longer produce warnings. Log: `docs/_build/solving-update.log`.
- Checked **116 local page and anchor links** and **three image references**; all resolve. Ruff lint, formatting, type checks, and `git diff --check` pass. No library code or figure scripts changed.

### Inference update

Updated on 2026-10-27. The sunny/high-bookings scenario now passes `observed={"weather": 1, "bookings": 1}` together with the fitted policy and obtains 50,000 conditional profit draws directly. The hotel-report information arc and refitting example remain, along with the original weather-based policy, fixed-order comparison, and sunny-day stockout question. The page explains aligned draws and support for fitted rules with continuous or mixed information. The inference and policy API pages now describe the same support boundary.

- All **nine code cells** execute successfully, with no notebook errors or inference-page warnings. Verified the displayed policy probabilities and numerical outputs against the executed notebook.
- The original policy's independently estimated expected profit is **626.50**, and the inference draws estimate **631.16**. For sunny days with high bookings, the refitted rule assigns about **70%** probability to ordering **260** ice creams and the conditional mean profit is **1309.28**. The fixed-order comparison estimates **562.20**, sunny-day demand averages **150.44**, and the stockout probability is **43.0%**. Prose and figure labels use the corresponding rounded values.
- Regenerated and visually inspected all four inference figures. The recommended-profit and sunny/high-bookings images changed. The comparison and sunny-demand images are unchanged. The plotting script now reads conditional profit draws from their `Posterior` object. Log: `docs/_build/inference-conditioning-figures.log`.
- The strict full build with forced notebook execution exits with code **1** and **four warnings**, all from the pending Multiple decisions and Count distributions notebook failures and their traceback reports. Log: `docs/_build/inference-conditioning-build.log`. The subsequent incremental build of the updated API references succeeds without warnings. Log: `docs/_build/inference-conditioning-api-build.log`.
- Checked **113 local links and anchors** and **seven image references** in the rendered inference page. All resolve.
- Full test suite: **410 passed**. Ruff lint, formatting, type checks, and `git diff --check` pass. Test log: `docs/_build/inference-conditioning-tests.log`. No library code changed.

### Solving training-budget advice

Added on 2026-10-27. Expanded the existing reliability advice into an info box distinguishing samples per training update, number of updates, and independent random keys. It recommends comparing fitted action probabilities and evaluating candidate policies on fresh scenarios, while distinguishing evaluation precision from policy improvement. The advice is general and includes no results from the separate training-budget experiments.

The Solving guide executes without errors, and the info box and its API and guide links render correctly. The full build retains the same four warnings from Multiple decisions and Count distributions. Log: `docs/_build/solving-training-advice-build.log`. No example code or library code changed.

### Multiple decisions update

Completed on 2026-10-27 without committing. Both decisions declare `TabularSoftmax()`, and the page uses the default fitting budget and current `Solution.render()` interface. It explains joint fitting, parentless promotion probabilities, conditional order probabilities, and the limited training received by rarely used skipped-promotion rows. The promotion story, demand model, separate utility nodes, and explicit-memory lesson are preserved.

- All **five code cells** execute successfully, both directly and through Sphinx. The fitted rules favour running the promotion and ordering **100** ice creams on cloudy days or **320** on sunny days when it runs. The independent total-utility estimate is **671.91**, and the separate inference estimate is **677.5**. Prose describes the stochastic rules and rounded probabilities rather than promising deterministic actions.
- Replaced the removed `is_solvable` and automatic solver-selection example with an actual fit of the diagram without the memory arc. Its order rule receives only `weather`, and its estimated total utility is **670.04**. The page explains why similar fitted recommendations do not remove the information restriction.
- Total-profit draws now add the two utility arrays by sample index before computing their mean. This preserves the relationship between sales profit and promotion cost in each simulated day.
- The strict full build with fresh Sphinx processing and forced notebook execution exits with code **1** and **two warnings**, both from the pending Count distributions notebook failure and its traceback report. The Multiple decisions page executes without warnings. Logs: `docs/_build/multiple-decisions-update.log` and `docs/_build/multiple-decisions-example-check.log`.
- Checked **107 local links and anchors** and **two image references** in the rendered page. All resolve. No new figure assets were needed.
- Ruff lint, formatting, type checks, and `git diff --check` pass. No library code or figure scripts changed. Runtime verification used the direct and Sphinx notebook executions.

### Common pitfalls update

Updated on 2026-10-27. Retained the table-indexing and parent-effect sections, checked their upstream references, and added three focused sections: mapping categorical state indices to quantities, combining related variables from one inference call, and evaluating the deterministic policy obtained by taking the most likely actions. Each section supplies its own imports and context.

- All **six code cells** execute successfully through direct execution and Sphinx. The categorical-index example maps action **1** to stock **160** and computes profit **1040.0**. The joint-draw example estimates a measurement-error standard deviation of **0.49** for a model configured with **0.5**. The deliberately short fitted-policy example gives action probabilities about **0.128** and **0.872**, with estimated payoff **1.73**, compared with **2.0** for the fixed most likely action.
- Each of the five sections also executes in a separate namespace, confirming that the examples do not depend on imports or variables from another section.
- The full strict build with forced notebook execution exits with code **1** and the same **two warnings**, both from the pending Count distributions notebook failure and its traceback report. Common pitfalls produces no Sphinx warnings. Logs: `docs/_build/common-pitfalls-update.log` and `docs/_build/common-pitfalls-example-check.log`.
- Checked **112 local links and anchors** and **one image reference** in the rendered page. All resolve.
- Ruff lint, formatting, type checks, and `git diff --check` pass. No library code, figure scripts, or image assets changed. Runtime verification used direct section execution and Sphinx notebook execution.

### Count distributions update

Completed on 2026-10-27 without committing. The capacity decision now declares `TabularSoftmax()`, and the example uses the current `Solution.render()` interface. Added a variant that chooses capacity after observing the customer count, using an explicit information arc and `LinearSoftmax`. Its fitted policy is evaluated with an observed Poisson count. The page separates policy-family information requirements from the enumeration requirements for counts left unknown during posterior inference.

- All **seven code cells** execute successfully, both directly and through Sphinx. The original policy favours capacity **20**, with probability about **0.999**, and estimates expected profit at **39.99**. The count-informed policy favours capacity **10** for **8** waiting customers and **20** for **16**. Evaluating the latter queue gives a rounded mean profit of **60.0**.
- The sampling example estimates mean arrivals at **12.0541** and the probability of at least twenty arrivals at **0.0238**. The rate example gives a posterior mean of about **2.524**, and the alarm example gives a mean defect count of **3.0672**. Existing numerical-count, fixed-support, and unsupported-domain explanations are preserved.
- The strict full build with fresh Sphinx processing and forced execution succeeds with **no Sphinx warnings**. All published notebooks execute without errors. Logs: `docs/_build/count-distributions-update.log` and `docs/_build/count-distributions-example-check.log`.
- Checked **118 local links and anchors** and **one image reference** in the rendered page. All resolve.
- Ruff lint, formatting, type checks, and `git diff --check` pass. No library code, figure scripts, or image assets changed. Runtime verification used direct page execution and Sphinx notebook execution.

### README, site overview, and final verification

Completed on 2026-10-27, with the final changes left uncommitted for review. The README now declares `TabularSoftmax`, uses the default training budget and `Solution.render()`, and explains fitted action probabilities. Both overview pages describe numerical decision information and distinguish policy quality from estimation noise. The Solving guide's remaining claim that a fitted recommendation was the best action has been corrected.

- The README example executes successfully. It estimates expected profit at **600.56**, with probabilities about **0.994** for ordering **80** on cloudy days and **0.991** for ordering **200** on sunny days. These match the quickstart's Bayesian-prior model. Log: `docs/_build/readme-example-check.log`.
- Checked the Python requirement and CUDA extras against `pyproject.toml`, and verified the platform and driver guidance against the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html). The installation page retains its supported commands and Graphviz guidance. Its wording and the README now agree about Linux NVIDIA support and experimental WSL2 support.
- Re-ran all **five figure scripts**, regenerating **nine published charts**. Their parameters match the pages, the plots were visually inspected, and the image files are unchanged. The quickstart policy script and inference script obtain their models directly from the corresponding page's code cells. Logs: `docs/_build/final-quickstart-figures.log`, `docs/_build/final-inference-figures.log`, and `docs/_build/final-density-figures.log`.
- The final strict HTML build uses fresh Sphinx processing and forced notebook execution and succeeds with **no Sphinx warnings**. Checked all **64 code cells across eight published notebooks**, with no unexecuted cells or error outputs. Old generated pages and notebooks were cleared before final verification so excluded drafts were not mistaken for current output. Log: `docs/_build/final-docs-build-clean.log`.
- Across **36 generated HTML pages**, all **4,407 local links and anchors** and **58 image references** resolve. The **four visible README documentation links** also point to pages present in the generated site. Log: `docs/_build/final-local-links.log`.
- The strict Sphinx external-link check succeeds: **38 destinations work directly** and **two redirect successfully**. Its internal fragment links are covered by the local check. All **18 visible README link and image destinations** respond successfully. The commented-out tutorial link remains unpublished. Logs: `docs/_build/final-external-links-clean.log` and `docs/_build/final-readme-links.log`.
- Searches through published pages, docstrings, and figure scripts found no remaining removed solver names, solver-selection examples, obsolete rendering calls, or deterministic fitted-policy iteration. Reviewed the excluded tutorial drafts again and confirmed the separate backlog above. They remain excluded, as do the superseded guide drafts.
- Ruff lint, formatting (**113 files**), type checks, and `git diff --check` pass. All **410 existing tests pass** in **62.49 seconds**. No tests or library changes were added in this final pass. Test log: `docs/_build/final-tests.log`.

## Next step

Review the uncommitted final documentation changes. The published-site migration has no remaining checklist items. Commit and update the documentation PR when requested. Rewriting and publishing the excluded tutorial drafts remains separate future work.
