# Test organization

Test directories follow the source package boundaries. Choose a location by the behavior under test; the imports needed to construct a diagram do not determine ownership. Descriptive filenames within each directory can cover several related behaviors without requiring one test file per source module.

| Directory | Responsibility |
| --- | --- |
| `tests/graph/nodes/` | Node construction, mutation, consistency, and declared policy families |
| `tests/graph/diagram/` | Diagram construction, structural validation, snapshots, and probing |
| `tests/graph/render/` | Shared rendering scenes, Mermaid, and Graphviz output |
| `tests/graph/test_action_space.py` | Scalar action-domain declarations |
| `tests/policy/_categorical/` | Fitted categorical rule methods, information encoding, and action probabilities |
| `tests/policy/_continuous/` | Distribution families, constraint transforms, continuous rules, and batched execution |
| `tests/policy/test_models.py` | Parameter-model output names and custom nested parameter trees |
| `tests/inference/` | Public posterior queries, conditioning, pruning, and posterior summaries |
| `tests/solver/` | Training, initialization, gradients, optimization settings, and analytic utility checks |
| `tests/_numpyro/` | Model translation, sampling, policy execution, and fitted metadata validation |
| `tests/integration/` | Distribution, count, and function-policy scenarios spanning execution layers |
| `tests/test_smoke.py` | Package version and public exports |

The NumPyro tests separate model construction in `test_model.py`, forward and posterior draws in `test_samplers.py`, and policy execution in `test_policies.py`. A test that calls `solve` to verify family initialization belongs with the solver; a test that directly inspects a fitted rule's probabilities belongs with policies. Cross-package scenarios remain together when exercising several execution paths is the purpose of the test.

Shared node validation contracts live in `tests/graph/nodes/test_node.py`; type-specific modules cover their distinct configuration and mutation behavior. Exact marginal-probability expectations cover normalization, while inference tests check the public result types, shapes, and posterior values.

Keep helpers local while only one module needs them. Shared policy scenarios live in `tests/_diagrams.py`; shared discrete backend diagrams live in `tests/_numpyro/diagrams.py`. These builders return fresh diagrams so tests can mutate them independently. Test modules import helpers rather than importing other test modules. Use `conftest.py` for pytest fixtures with an actual shared lifecycle or parametrization need.

Runnable examples and regression tests are independent: neither imports the other. The oil benchmark's diagram, probability tables, and exact utility evaluator live locally in `tests/solver/test_oil_policy.py`. Its small scenario is intentionally duplicated in `examples/oil_policy.py` so changes to the demonstration cannot change the regression reference.

Run the complete suite with `uv run pytest tests/`, or pass a package directory to focus on one area. Pytest discovery and CI use the `tests/` root, so directory moves do not require separate configuration.
