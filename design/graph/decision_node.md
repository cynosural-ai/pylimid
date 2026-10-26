# Decision nodes

A `DecisionNode` specifies a controlled variable with a finite action menu, the information available when it acts, and an optional trainable policy family. It has no chance distribution. All decisions share this representation.

- `states` labels the available actions. Values passed to downstream functions are integer indices into that tuple. `states=None` means the node is unconfigured; continuous actions are not yet supported.
- `parents` names exactly the observed variables. Earlier actions and observations are remembered only through explicit arcs. Information arcs do not represent chance distributions.
- `policy_family` declares a `PolicyFamily`, such as `LinearSoftmax` or `TabularSoftmax`, for optimization. The creator chooses the family, and `solve` fits its parameters jointly with the other decisions. A missing family is allowed for supplied-policy evaluation, but not for optimization.

The decision is structurally configured once its action states are declared. Policy-family compatibility is checked by the optimizer using the actual sampled information shapes and declared categorical labels. Adding an information arc does not change the action space.

`LinearSoftmax` accepts numerical or mixed information; `TabularSoftmax` requires scalar categorical information. Fitted policies preserve their local information binding and stochastic behavior when reused for forward evaluation. Supplied deterministic tables and functions remain supported by inference.

The [optimizer design](../solver/algorithms.md) describes the expected-utility objective, score estimator, and verification. The [policy notes](../continuous_information/README.md) record representation limits and future extensions. [Diagram validation](diagram.md) covers structural checks, consistency, and snapshots.
