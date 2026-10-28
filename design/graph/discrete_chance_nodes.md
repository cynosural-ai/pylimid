# Categorical chance nodes and numerical counts

Chance nodes distinguish category codes from numerical values. A weather state such as `sunny` has a label and an integer code, while a count of defective products has numerical meaning. Both may be mathematically discrete, but their summaries and policy-information requirements differ.

## Value semantics

A chance node with `states` declares named categories. Its callable must return Categorical with one label per category, or Bernoulli with exactly two labels. Values passed to downstream functions are indices into that tuple. Probability and logit parameterizations are supported, and zero-probability categories still require labels.

A chance node with `states=None` represents numerical values. Continuous distributions and scalar-event count distributions such as Binomial and Poisson retain their actual draws. A count of two means two items, not an index into an implicit label table. Numerical counts cannot acquire categorical semantics by attaching string labels to their distribution.

`ChanceNode.is_categorical` and `Posterior.is_categorical` describe these value semantics. Mathematical discreteness is determined from the constructed distribution during execution. The graph does not run callables to classify incomplete nodes, infer support from sampled dtypes, or store runtime distribution properties on mutable nodes.

The [chance-node implementation](../../pylimid/graph/nodes/chance_node.py) owns the declarations. The [shared distribution validator](../../pylimid/_numpyro/distributions.py) checks Categorical and Bernoulli cardinalities, rejects labels on numerical count distributions, and rejects vector discrete events such as Multinomial. Batched scalar counts retain their array shape in forward draws.

## Requirements belong to operations

| Operation | Current behavior |
| --- | --- |
| Forward simulation | Numerical counts can feed downstream distributions and utilities without a finite list of labels. |
| Policy-gradient training | Counts can be sampled as outcomes or supplied as numerical information to compatible policy families. |
| Observed count likelihood | Observed counts contribute their distribution's likelihood without requiring enumeration of that observed node. |
| Unknown count in a posterior query | Requires scalar, enumerable, fixed zero-based integer support. |
| Categorical policy tables | Information parents require named categorical states. Numerical counts do not supply table indices. |

A model can therefore support simulation and solving while a particular posterior query remains unsupported. For example, Poisson arrivals can determine a payoff or be observed while inferring an unknown arrival rate. An unknown Poisson count needed by a posterior query is rejected rather than treated as a continuous NUTS variable or silently truncated.

`InfluenceDiagram.validate()` remains structural and declarative. Inference prunes nodes unrelated to the queries and evidence, then checks constructed distributions in a seeded model execution. Observed and deterministic sites do not count as unknown latents. The [model translator](../../pylimid/_numpyro/model.py) annotates enumerable discrete sites, and the [sampling dispatcher](../../pylimid/_numpyro/samplers.py) checks posterior capabilities before choosing enumeration alone or enumeration with continuous NUTS.

## Fixed-range count posteriors

Unknown numerical counts require a scalar distribution with NumPyro enumeration support and an integer interval starting at zero. Fixed-trial Binomial and Beta-Binomial distributions satisfy this contract. Their probability parameters may depend on categorical or continuous parents while their range stays fixed.

The enumeration bound must remain concrete during tracing. A Python integer `total_count` works, while a bound constructed with JAX inside the callable can fail even when mathematically constant. NumPyro's errors for non-static or heterogeneous bounds propagate directly. PyLIMID does not reinterpret arbitrary tracing errors from a user's callable as count-domain errors.

A range containing only zero is clamped to that exact value because NumPyro cannot reconstruct the singleton enumerated site. Shifted ranges are rejected because the backend can return enumeration positions instead of original count values. Batched count latents and unbounded ranges are also unsupported. Neither unbounded support nor a large finite range is silently truncated.

Parent-dependent trial counts support forward simulation but not posterior enumeration. A factory can simulate defects in a batch containing either five or ten products. Conditioning on a downstream alarm still requires enumerating the unknown defect count, and the varying trial count fails NumPyro's enumeration requirements. A mathematically finite union of possible outcomes does not establish a supported execution path.

## Counts as decision information

`LinearSoftmax` and the `Linear` parameter model consume counts as numerical features. They do not require a finite count domain. Each fitted rule receives only its declared information parents, using the same binding and feature encoding as other numerical observations.

`TabularSoftmax` and supplied categorical action tables require scalar categorical information with named states. A numerical count cannot be used directly as a policy-table index. A model can introduce a separate categorical report, such as low or high arrivals, when the decision observes categories rather than the exact count.

The [policy notes](../continuous_information/README.md) describe the families and feature encoding, and [policy binding](../../pylimid/policy/_binding.py) checks compatibility with the diagram's declarations.

## Posterior summaries

`Posterior` retains raw JAX draws with their shape, dtype and joint sample order. Category probabilities use `marginal()`, and numerical summaries reject arbitrary category codes. Counts use `mean()`, `std()` and `hdi()` like other numerical variables.

A count interval may contain values absent from the draws and need not be a smallest credible set. Event probabilities such as `P(arrivals >= 20)` can be estimated directly from the draws. `marginal()` requires named categories and does not return a probability vector over only the counts that happened to appear in a sample.

The [posterior implementation](../../pylimid/inference/posterior.py) owns these summary contracts. Observations use category indices for categorical nodes and actual numerical values for counts.

## Verification

- [Distribution checks](../../tests/test_distributions.py) cover categorical cardinalities, numerical count declarations and discrete event shapes.
- [Forward count tests](../../tests/test_counts.py) cover numerical values, downstream distributions and utilities, observed-count likelihoods and rejection of count-indexed policy tables.
- [Bounded-count posterior tests](../../tests/inference/test_bounded_counts.py) compare Binomial and Beta-Binomial masses and moments with explicit probability sums. They cover categorical-parent updates, mixed continuous/discrete posteriors, joint utility draws, zero-trial counts and rejection of unsupported count domains.
- [Posterior tests](../../tests/inference/test_posterior.py) cover categorical versus numerical summaries, array shapes and numerical intervals.
