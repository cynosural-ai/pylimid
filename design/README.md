# PyLIMID design notes

Living contributor documentation of how PyLIMID works and why. These notes describe the current implementation, its tradeoffs and supported boundaries, with links to relevant code and tests. Update the relevant note when the implementation changes. Published user documentation lives in [`docs/`](../docs/), and concrete remaining work lives in [`TODO.md`](../TODO.md).

## Current design

Read the graph principles first, then the notes relevant to the change:

- [Diagram and validation](graph/diagram.md): mutable construction, consistency checks, and snapshots.
- [Chance nodes](graph/chance_node.md): callable distributions, state indices, and conditional probability tables.
- [Categorical chance nodes and numerical counts](graph/discrete_chance_nodes.md): value semantics, runtime distribution checks, and operation-specific inference limits.
- [Decision nodes](graph/decision_node.md): explicit information sets, categorical and scalar continuous actions, and declared policy families.
- [Utility nodes](graph/utility_node.md): callable payoffs, sink validation, and additive aggregation.
- [NumPyro translation and sampling](inference/backend_numpyro.md): model construction, policy binding, and posterior draws.
- [Policy-gradient optimization](solver/algorithms.md): joint expected-utility training, local information, independent evaluation, and analytic verification.
- [Policy families and mixed information](continuous_information/README.md): categorical and continuous families, fitted rules, and custom parameter models.
- [Posterior inference with fitted policies](inference/fitted_policy_conditioning.md): conditioning through categorical and continuous action distributions, with mixed decision information.
- [Test organization](testing.md): package ownership, integration scenarios, and shared test helpers.

These notes explain the reasoning behind the current implementation. Public signatures and behavior belong in the API reference and docstrings rather than a second implementation-status table here.

## Documentation work

Read the [documentation maintenance guide](documentation_plan.md) before changing published pages or the project README. It records the site structure, writing conventions, maintenance checks and future additions.
