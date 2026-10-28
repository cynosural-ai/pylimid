# Decision nodes

A `DecisionNode` specifies a controlled variable with an action domain, the information available when it acts, and an optional trainable policy family. It has no chance distribution. Categorical and continuous decisions share this representation.

- `states` labels categorical actions. Values passed to downstream functions are integer indices into that tuple.
- `action_space` declares a scalar continuous domain through `Real()` or `Interval(low, high)`. The immutable domains are owned by `pylimid.graph.action_space` and exported by `pylimid.graph` and `pylimid`. Interval bounds are fixed finite real numbers with `low < high`, including the endpoints. Domain declarations do not select a distribution or policy structure.
- `parents` names exactly the observed variables. Earlier actions and observations are remembered only through explicit arcs. Information arcs do not represent chance distributions.
- `policy_family` declares a `PolicyFamily`, such as `LinearSoftmax` or `TabularSoftmax`, for optimization. The creator chooses the family, and `solve` fits its parameters jointly with the other decisions. A missing family is allowed for supplied-policy evaluation, but not for optimization.

The decision is structurally configured once exactly one of `states` or `action_space` is declared. Neither means `UNCONFIGURED`. Both means `STALE`, reported through ordinary node and diagram validation. Field assignments remain independent so an author can switch representations incrementally. Adding an information arc does not change the action space.

Continuous decisions use `DistributionPolicy` or a custom family returning a scalar continuous density with fixed support contained in the declared domain. The common score-function solver fits categorical and continuous rules jointly. `ContinuousPolicy` retains the fitted family, parameters, local information metadata, and action domain. It exposes distribution inspection, sampling and log densities. Inference accepts these rules for forward sampling and conditioning, and also accepts fixed real actions and deterministic real-valued functions for forward evaluation. An unrelated continuous decision is pruned as usual.

`DistributionPolicy` composes a NumPyro constructor and a `ParameterModel`. `Linear(outputs=None)` discovers scalar parameter names from constructor metadata, excluding fixed parameters, and computes linear raw predictions from the existing feature encoding. NumPyro constraint bijections convert them into valid distribution parameters. TruncatedNormal requires explicit fixed bounds. No action clipping, automatic truncation or automatic scaling changes the creator's selected distribution. Vector actions, parameter-dependent supports and constructors with unsupported metadata remain outside automatic composition.

For categorical decisions, policy-family compatibility is checked by the optimizer using the actual sampled information shapes and declared categorical labels.

`LinearSoftmax` accepts numerical or mixed information; `TabularSoftmax` requires scalar categorical information. Fitted policies preserve their local information binding and stochastic behavior when reused for forward evaluation. Supplied deterministic tables and functions remain supported by inference.

The [optimizer design](../solver/algorithms.md) describes the expected-utility objective, score estimator, and verification. The [policy notes](../continuous_information/README.md) cover representation limits and custom families. [Diagram validation](diagram.md) covers structural checks, consistency, and snapshots.
