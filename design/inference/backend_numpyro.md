# NumPyro translation and sampling

The internal bridge between the graph layer and NumPyro lives in `pylimid._numpyro`. Its modules expose shared operations to other pylimid modules; the package does not expose a supported user API. The [graph principles](../graph/diagram.md) settle the container and callable convention, and this note covers execution.

NumPyro is the only probabilistic execution backend. Keeping one backend avoids parallel distribution extraction, capability checks and result contracts for separate exact engines. The same user-supplied distribution callables serve discrete and continuous models, while analytic expectations and small-model exact arithmetic provide independent test references.

## Translation

`build_model(snapshot, observed, policy_rules)` consumes a validated `DiagramSnapshot`, observed values, and already-bound decision rules. It captures the nodes in topological order. The returned model resolves parent values by name and handles each node as follows:

- A chance node constructs its distribution with `node.dist(**parent_values)` and emits a `numpyro.sample` site. Observed values are clamped; unobserved discrete sites carry parallel-enumeration metadata.
- A fixed decision is a degenerate observed site at its chosen categorical index or real action value.
- A supplied table or deterministic function selects an action using its realized information-set values and emits a deterministic site. A fitted categorical or continuous rule emits a sample site with its local action distribution. Only categorical actions receive enumeration metadata.
- A utility node is skipped by the translator. When a utility is queried, the sampling wrapper evaluates its callable on each joint draw of its parents.

`infer()` in `pylimid.inference.api` validates the whole diagram, then retains the ancestors of the queries and evidence. Only decisions that influence those nodes need a binding. It separates fixed actions from supplied policies before calling the sampling layer. `bind_policy_rules` in `pylimid.policy._binding` dispatches fitted rules, deterministic functions, and action tables to separate binding helpers. Policy tables use integer state indices in the decision's parent order, with validation and array construction in `pylimid.policy._categorical.action_table`. The public `pylimid.policy` package exposes graph-independent families and fitted rules without eagerly importing binding helpers. NumPyro distribution checks live in `pylimid._numpyro.distributions` and are shared by inference and optimization.

Policy implementations are grouped by action type in the internal `pylimid.policy._categorical` and `pylimid.policy._continuous` packages. Each package keeps its fitted rule in `policy.py`, alongside its family implementations and type-specific checks. The shared family interface in `family.py`, parameter models in `models.py`, joint-policy aliases in `types.py`, and information handling in `_information.py` stay at the policy package level. Public imports remain available through `pylimid.policy` and `pylimid`.

The sampling layer binds supplied policies once and passes the resulting rules to the model builder. User-facing queries go through `infer`; model translation and raw sampling remain internal operations. The backend package initializer does not re-export those operations.

## Sampling and results

`samples()` wraps forward and posterior sampling:

- Without chance evidence, it draws directly from the model.
- With chance evidence and only discrete latent variables, it draws from the posterior using discrete enumeration.
- With chance evidence and continuous latent variables, including fitted continuous actions, NUTS samples them with discrete latents enumerated out. A conditional pass recovers discrete draws given the continuous draws.

Queried utilities are computed from their parents' draws. `infer()` wraps requested arrays in `Posterior` objects, retaining joint sample order and JAX array shapes and dtypes. Category probabilities are computed by `Posterior.marginal()`, while numerical variables and utilities use `mean()`, `std()`, and `hdi()`. Results are Monte Carlo estimates by contract, including when enumeration is used to obtain the draws.

One result type retains raw draws for every variable rather than selecting a summary representation during inference. Means and standard deviations cannot describe every numerical posterior, and categorical state indices have no intrinsic numerical meaning. `Posterior` therefore rejects numerical summaries of categories and rejects `marginal()` for numerical variables, including counts. Callers can inspect raw draws when they deliberately need another statistic. The [posterior tests](../../tests/inference/test_posterior.py) cover these summary contracts, shapes and category checks.

Inference tests use analytic posterior expectations and small-model exact arithmetic. The optimizer uses a joint policy-gradient trajectory walk in `pylimid.solver`; the [optimizer note](../solver/algorithms.md) covers its estimator and verification. There are no external reference-solver dependencies in the test suite.

## Constraints

**Index probability tables in one operation.** Under parallel enumeration, parents may carry enumeration dimensions. Use `T[a, b]` rather than `T[a][b]`; chained indexing can yield incorrect posterior weights or an enumeration error. The upstream issue is [NumPyro #2252](https://github.com/pyro-ppl/numpyro/issues/2252); revisit this limitation when it is resolved.

**Snapshots are logical views.** A `DiagramSnapshot` holds references to mutable nodes. The translator captures the node sequence, but later changes to those nodes can still affect execution. Callers must not mutate captured nodes before execution; see the [snapshot contract](../graph/diagram.md#principle-7--the-snapshot-is-logical-not-deep-frozen).

**The graph stays independent of NumPyro.** NumPyro and JAX are runtime dependencies, but `pylimid.graph` works with opaque distribution callables and does not import NumPyro. Execution-specific interpretation belongs in this layer rather than the graph container.
