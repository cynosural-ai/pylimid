# PyLIMID design notes

Contributor-facing rationale and proposals. Published documentation lives in [`docs/`](../docs/); the forward plan and known issues live in [`TODO.md`](../TODO.md).

## Current design

Read the graph principles first, then the notes relevant to the change:

- [Diagram and validation](graph/diagram.md): mutable construction, consistency checks, and snapshots.
- [Chance nodes](graph/chance_node.md): callable distributions, state indices, and deferred table sugar.
- [Decision nodes](graph/decision_node.md): explicit information sets, finite actions, and future policy optimization.
- [Utility nodes](graph/utility_node.md): callable payoffs, sink validation, and additive aggregation.
- [NumPyro translation and sampling](inference/backend_numpyro.md): model construction, policy binding, and posterior draws.
- [Solver algorithms](solver/algorithms.md): both scans, backward induction, validation references, and deferred single policy updating.
- [Backward-induction estimator](solver/backward_induction.md): conditional value estimates, sampling coverage, and the solvability gate.

These notes explain the reasoning behind the current implementation. Public signatures and behavior belong in the API reference and docstrings rather than a second implementation-status table here.

## Proposals

- [Discrete decisions with continuous information](continuous_information/README.md): creator-selected policy families, solver-fitted parameters, and one policy abstraction for tables and functions, independently of continuous-action support.
- [More expressive discrete chance nodes](graph/discrete_chance_nodes.md): a future proposal separating categorical and numerical semantics, finite computational domains, and display labels.
- [Ride-hailing retention](proposals/ride_hailing.md): a future benchmark for continuous decisions and learning under uncertainty.

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
