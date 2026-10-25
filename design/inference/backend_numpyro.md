# NumPyro translation and sampling

The bridge between the backend-agnostic graph layer and NumPyro lives in `pylimid.numpyro`. The [graph principles](../graph/diagram.md) settle the container and callable convention; this note covers execution. The [NumPyro-only ADR](../ADR/25_08_2026_numpyro_only_engine.md) records why the exact engines were retired.

## Translation

`to_model(snapshot, observed, policy)` consumes a validated `DiagramSnapshot` and captures its nodes in topological order. The returned model resolves parent values by name and handles each node as follows:

- A chance node constructs its distribution with `node.dist(**parent_values)` and emits a `numpyro.sample` site. Observed values are clamped; unobserved discrete sites carry parallel-enumeration metadata.
- A fixed decision is a degenerate observed site at its chosen action index.
- A contingent decision looks up its action from the policy table using its realized information-set values and emits a deterministic site.
- A utility node is skipped by the translator. When a utility is queried, the sampling wrapper evaluates its callable on each joint draw of its parents.

`infer()` in `pylimid.inference.api` validates the whole diagram, then retains the ancestors of the queries and evidence. Only decisions that influence those nodes need a binding. It separates fixed actions from contingent tables before calling the sampling layer. Policy tables use integer state indices in the decision's parent order; their validation and array construction are shared through `pylimid.policy`.

The translator constructs a model; the caller chooses how to execute it. For a Bayesian network:

```python
import jax
from numpyro.infer import Predictive
from pylimid.numpyro import to_model

model = to_model(diagram.snapshot())
prior = Predictive(model, num_samples=10_000)(jax.random.PRNGKey(0))
```

## Sampling and results

`samples()` wraps forward and posterior sampling:

- Without chance evidence, it draws directly from the model.
- With chance evidence and only discrete latent variables, it draws from the posterior using discrete enumeration.
- With continuous latent variables, NUTS samples them with discrete latents enumerated out; a conditional pass recovers discrete draws given the continuous draws.

Queried utilities are computed from their parents' draws. `infer()` wraps requested arrays in `Posterior` objects, retaining joint sample order and JAX array shapes and dtypes. Discrete probabilities are computed by `Posterior.marginal()`; continuous and utility summaries use `mean()`, `std()`, and `hdi()`. Results are Monte Carlo estimates by contract, including when enumeration is used to obtain the draws.

Inference tests use analytic posterior expectations and small-model exact arithmetic. The optimizer uses a joint policy-gradient trajectory walk in `pylimid.solver`; the [optimizer note](../solver/algorithms.md) covers its estimator and verification. There are no external reference-solver dependencies in the test suite.

## Constraints

**Index probability tables in one operation.** Under parallel enumeration, parents may carry enumeration dimensions. Use `T[a, b]` rather than `T[a][b]`; chained indexing can yield incorrect posterior weights or an enumeration error. The upstream issue is [NumPyro #2252](https://github.com/pyro-ppl/numpyro/issues/2252); revisit this limitation when it is resolved.

**Snapshots are logical views.** A `DiagramSnapshot` holds references to mutable nodes. The translator captures the node sequence, but later changes to those nodes can still affect execution. Callers must not mutate captured nodes before execution; see the [snapshot contract](../graph/diagram.md#principle-7--the-snapshot-is-logical-not-deep-frozen).

**The graph stays independent of NumPyro.** NumPyro and JAX are runtime dependencies, but `pylimid.graph` works with opaque distribution callables and does not import NumPyro. Execution-specific interpretation belongs in this layer rather than the graph container.
