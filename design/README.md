# PyLIMID design notes

Contributor-facing rationale and proposals. Published documentation lives in [`docs/`](../docs/); the forward plan and known issues live in [`TODO.md`](../TODO.md).

## Current design

Read the graph principles first, then the notes relevant to the change:

- [Diagram and validation](graph/diagram.md): mutable construction, consistency checks, and snapshots.
- [Chance nodes](graph/chance_node.md): callable distributions, state indices, and deferred table sugar.
- [Decision nodes](graph/decision_node.md): explicit information sets, categorical and scalar continuous actions, and declared policy families.
- [Utility nodes](graph/utility_node.md): callable payoffs, sink validation, and additive aggregation.
- [NumPyro translation and sampling](inference/backend_numpyro.md): model construction, policy binding, and posterior draws.
- [Policy-gradient optimization](solver/algorithms.md): joint expected-utility training, local information, independent evaluation, and analytic verification.
- [Policy families and mixed information](continuous_information/README.md): categorical and continuous families, fitted rules, and custom parameter models.
- [Posterior inference with fitted policies](inference/fitted_policy_conditioning.md): conditioning through categorical and continuous action distributions, with mixed decision information.

These notes explain the reasoning behind the current implementation. Public signatures and behavior belong in the API reference and docstrings rather than a second implementation-status table here.

## Proposals

- [More expressive discrete chance nodes](graph/discrete_chance_nodes.md): implemented count support and remaining proposals for finite numerical domains, display labels, and broader posterior inference.
- [Ride-hailing retention](proposals/ride_hailing.md): a future benchmark for continuous decisions and learning under uncertainty.

LLM integration remains an unscoped idea, potentially in a separate repository. Revisit its scope and documentation after the main pages, API review, and tutorials are ready.

## Documentation work

Read the [documentation maintenance guide](documentation_plan.md) before changing published pages or the project README. It records the site structure, writing conventions, maintenance checks and future additions.

## Decision records

ADRs preserve decisions in their dated context. Historical API names and package layouts may differ from today's implementation; the current-design notes above are the starting point for changes.

- [2026-07-16: initial vision](ADR/16_07_2026%20-%20initial%20plan.md).
- [2026-07-17: mutable workspace](ADR/17_07_2026_mutability_design_decision.md).
- [2026-07-23: unified inference architecture](ADR/23_07_2026_unified_inference_architecture.md), partly superseded by the NumPyro-only decision. The current package boundaries are recorded in the [contributor guide](../CONTRIBUTING.md#repository-layout).
- [2026-08-13: typed inference results](ADR/13_08_2026_typed_inference_results.md), superseded by the unified posterior.
- [2026-08-25: NumPyro-only engine](ADR/25_08_2026_numpyro_only_engine.md).
- [2026-08-25: unified posterior](ADR/25_08_2026_unified_posterior_result.md).
- [2026-10-09: structural checks in validation](ADR/09_10_2026_structural_checks_in_validate.md).

Retired engine plans and completed example plans remain recoverable from Git history.
