# More expressive discrete chance nodes

Status: classification groundwork, forward count simulation and observed-count likelihoods are implemented. Posterior inference with latent numerical counts, numerical information-set domains and independent label metadata remain future work.

The classification step replaces the label-based `is_discrete` property on nodes and `Posterior` with `is_categorical`. For now, `states` continues to declare categorical semantics as well as labels. Model translation checks support and annotates only categorical sites for enumeration. Posterior dispatch inspects constructed distributions in a seeded model execution, ignores observed and deterministic sites, rejects all latent numerical discrete nodes and reports whether continuous latents remain. It does not evaluate callables with fabricated parent values, infer support from sampled dtypes, or store execution-derived classification on mutable graph nodes.

## Motivation

Chance nodes describe their distributions with a callable, which returns a NumPyro distribution based on parent values. Categorical outcomes declare a tuple of string `states`, and sampled values are interpreted as indices into that tuple. Numerical count distributions such as Binomial and Poisson use `states=None`; their draws retain actual numerical values.

That fits weather represented by `("cloudy", "sunny")`. It is less natural for a count of defective products, a number of arrivals, or an integer-valued measurement. These are discrete quantities with numerical meaning. Requiring labels such as `("0", "1", ..., "10")` duplicates a bounded count distribution's domain, and no finite tuple can describe every possible outcome of an unbounded count distribution.

The longer-term objective is to represent categories and numerical discrete variables directly, while making the requirements of each inference and solving method explicit. More expressive chance nodes should not imply that every distribution becomes supported by every operation.

## What `states` means today

The current implementation gives `states` several responsibilities:

| Responsibility | Current implementation |
| --- | --- |
| Declare categorical semantics | `ChanceNode.is_categorical` checks whether `states` is present, without executing the distribution. |
| Define a finite domain | Valid values are interpreted as indices from zero to `len(states) - 1`. |
| Supply display labels | `InfluenceDiagram.states_of` and `Solution.render` decode indices using the labels. |
| Size policy tables | Shared policy helpers and the solvers use the number of states of each information-set parent. |
| Select posterior inference machinery | Only categorical sites are marked for enumeration. Runtime support identifies numerical counts, which may be forward sampled or observed but cannot remain latent in posterior inference. |
| Control posterior summaries | `Posterior.is_categorical` distinguishes named categories from numerical values; `marginal()` requires categories, while numerical summaries reject category codes. |

The current runtime contract requires Categorical with one state label per category or Bernoulli with exactly two labels. Both probability and logit parameterizations are supported, and zero-probability outcomes still require labels. With `states=None`, continuous distributions and numerical discrete distributions with scalar events are supported. Batched scalar-event counts retain their array shape; vector discrete events such as Multinomial are rejected. Adding labels to Binomial, Poisson or another numerical count distribution is rejected.

The [shared distribution validator](../../pylimid/_distributions.py) checks the constructed distribution before sampling or emitting a NumPyro sample site. It runs in inference and all three solvers, including for observed chance nodes. It uses concrete categorical classes, static category dimensions, event shape and support metadata rather than sample probes. `InfluenceDiagram.validate()` remains structural and declarative; `infer()` prunes nodes unrelated to its query and evidence before executing distributions. A separate posterior capability check rejects relevant latent counts before either posterior sampling path runs. Tabular policies continue to require named finite states and cannot use count values as indices.

See [the chance-node implementation](../../pylimid/graph/chance_node.py), [policy helpers](../../pylimid/policy.py), [NumPyro translation](../../pylimid/numpyro/model.py), [sampling dispatch](../../pylimid/numpyro/samplers.py), and [posterior summaries](../../pylimid/inference/posterior.py).

These responsibilities need not share one field. In the proposed representation, an unlabeled variable could be discrete, and a discrete variable could have numerical meaning. A finite domain could support computation without requiring a string label for every outcome.

## Examples the representation should accommodate

The count examples below support forward simulation and solving when information sets remain categorical. Observed counts support likelihood evaluation. Latent numerical counts remain unsupported in posterior inference, including bounded counts.

### Categories

Weather has two named outcomes. The sampled integers are category codes, and their numerical distance has no meaning. A probability vector and readable labels are appropriate; the mean category code is not a useful summary.

Existing categorical models should keep their parent-callable convention: functions receive category indices, and probability or utility tables can be indexed by them. Renaming a label must not change the sampled codes or probabilities.

### Bounded numerical counts

The number of defective items in a batch of ten can follow `Binomial(total_count=10, probs=0.2)`. Its values are actual counts from zero through ten. An expected count of two is meaningful. Users should not need eleven strings to express this model.

The defect probability may depend on weather, a supplier category, or a continuous parameter. With a fixed trial count, the finite outcome range stays unchanged as those parameters vary.

### Unbounded numerical counts

Arrivals can follow a Poisson distribution. With numerical count support, a forward simulation could sample an arrival count and pass it to a utility function without enumerating every possible count.

This does not make an arrival count suitable as a direct parent of a finite policy table, or make an unobserved Poisson node compatible with the current posterior-enumeration path. Those operations need additional capabilities. A user who needs a decision to observe demand bands can construct a separate finite report, while preserving the underlying count variable.

### Parent-dependent bounds

A factory may process either five or ten items, selected by a `batch` parent:

```python
BATCH_SIZES = jnp.array([5, 10])


def defects_dist(batch):
    return dist.Binomial(total_count=BATCH_SIZES[batch], probs=0.2)
```

The conditional support is zero through five for one parent value and zero through ten for the other. A fixed global domain of zero through ten covers both. It is a computational envelope, not a claim that every outcome has positive probability in every situation.

Forward simulation of `batch -> defects -> alarm` is supported. Querying defects given an observed alarm raises an explicit unsupported-latent-count error before NumPyro enumeration. Earlier enumeration attempts failed with a JAX `ConcretizationTypeError` at `jnp.arange(total_count + 1)`; NumPyro also rejects enumeration with heterogeneous concrete trial counts. Any future extension must distinguish valid forward simulation from supported posterior enumeration. See [NumPyro's Binomial enumeration implementation](https://num.pyro.ai/en/stable/_modules/numpyro/distributions/discrete.html#BinomialProbs.enumerate_support) and [JAX's explanation of concrete-value requirements](https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError).

## Proposed separation of concerns

The conceptual model should distinguish four things. Exact field names and constructor syntax remain open.

| Concept | Meaning | Source |
| --- | --- | --- |
| Distribution support | Which values the conditional distribution permits, including whether they are discrete or continuous. | The constructed NumPyro distribution. |
| Value semantics | Whether integers are category codes or numerical quantities. | Explicit model semantics; a distribution class alone cannot always establish this. |
| Finite computational domain | A fixed domain available to operations that enumerate outcomes or construct policy tables. | Derived when reliably available, otherwise explicitly declared when required. |
| Display labels | Optional names for finite outcomes. | User-provided presentation metadata. |

Categorical and numeric semantics must be independent of label presence. Adding labels to numerical counts must not disable their mean. Removing labels from a categorical node must not turn its codes into a continuous variable or a meaningful numerical measurement. A Categorical distribution can also encode a numerical count distribution, so the class name is not sufficient to settle the variable's meaning.

Use NumPyro's existing support, enumeration, and shape metadata rather than implementing a parallel distribution system. That metadata describes the distribution's mathematical properties; it does not establish user-facing semantics or a global domain across every parent configuration. See [NumPyro's distribution interface](https://num.pyro.ai/en/stable/distributions.html#numpyro.distributions.distribution.Distribution).

Initially, numerical discrete distributions have scalar events; batched scalar counts retain component-wise summaries. Numerical information-set domains could start with contiguous integer ranges. Arbitrary sparse integer domains and vector-event discrete distributions are further capabilities, not requirements for this proposal. Existing array-valued continuous variables, including Dirichlet priors, retain their JAX draw and summary shapes.

## Requirements belong to operations

The following table describes the intended boundaries after this proposal, not capabilities already implemented everywhere.

| Operation | Requirement |
| --- | --- |
| Forward sampling of a chance variable | The distribution must support the simulation's JAX transformations and have a compatible output shape. Finite display labels are not intrinsically required. |
| Expected utility under a policy | Chance variables may be numerical counts without a finite domain if they are used numerically rather than as policy-table indices. The policy still needs valid finite information-set domains. |
| Tabular policy construction | Every observed parent needs a fixed finite domain and a checked mapping from its values to table indices. |
| Discrete posterior enumeration | Each enumerated latent needs support that the selected NumPyro enumeration path can handle. Mathematical finiteness alone is insufficient. |
| Continuous posterior sampling | Continuous latent variables use a compatible continuous sampler. An unlabeled discrete latent must never be dispatched to NUTS as though it were continuous. |
| Observed discrete likelihood | Observed counts need no latent enumeration of their own. Other latent variables and their inference requirements still determine whether the complete query is supported. |

Consequently, a model may support forward simulation and solving while a particular posterior query is unsupported. That difference should be documented and reported where the operation is requested. Unsupported inference should stop with an error, not trigger an unannounced approximation or a change of model.

The regularity check remains a structural question about backward-induction applicability. Finite domains and numerical sampling coverage are separate prerequisites; adding count support does not change the regularity theorem.

## Finite domains and policy indexing

Decisions remain finite action choices. This proposal does not introduce continuous actions or policies over infinite information sets.

The solvers need domain sizes before they enumerate candidate rules or allocate action tables. Because a node stores a callable, its distribution cannot always be inspected without parent values. Do not call arbitrary functions with zero-valued parents to infer a supposedly global domain, and do not derive it from the largest outcome seen in a preliminary sample.

An explicit finite domain should be required when the operation needs one and it cannot be established reliably. A declared global range can contain outcomes that are impossible for some parent configurations. It must never silently truncate or clamp a distribution. A Poisson variable does not become a bounded variable merely because the model declares a finite domain for it.

Keep actual numeric counts in samples, observed evidence, utilities, and downstream distributions. Convert a value to a dense table index only at the policy boundary, with a checked mapping. This also avoids assuming that a future finite range starting at three can be indexed directly by the count three.

Preserving the existing index-based `Policy` representation is a reasonable first implementation. A single domain mapping should encode numeric observations and decode rendered policy rows. Whether public policy keys should eventually use actual numeric values is a separate API decision; it must not change accidentally while replacing label lookups.

All solvers must share the domain interpretation, including backward-induction group indices and both scan implementations. Declaring a finite domain must not hide out-of-domain values through JAX's indexing behavior.

Larger domains still increase computational cost. Scan policy counts remain exponential in the number of information-set assignments, so automatically deriving a large finite domain does not make its policy search practical. Rare and impossible assignments also retain their existing sampling-coverage problems; domain representation alone does not solve those issues.

## Runtime metadata and graph validation

Keep `InfluenceDiagram.validate()` focused on structural and declarative consistency. It can validate labels, explicit domains, and declared semantics without pretending to prove the output of an arbitrary callable.

Execution paths already share the categorical contract check when constructing distributions. The broader implementation should extend this into a shared interpretation of support, shape, value semantics, and finite domains rather than rebuilding classification rules in the translator and each solver. Its first version can reject distributions whose mathematical kind or output shape changes with parent values, and domains whose compatibility cannot be established safely for an indexing operation.

Do not infer discreteness from the presence of labels or from the dtype of one sampled value. The graph-level `is_categorical` property describes declarations only. Mathematical discreteness is resolved from distributions during execution; incomplete nodes are not classified as continuous by default.

The graph layer need not import NumPyro to inspect a distribution it has not constructed. User declarations can remain on the nodes; execution-derived properties belong to the execution that resolved them. As diagrams are mutable, resolved domain information must not be written back as an indefinitely valid fact about the node or reused after relevant edits.

Let NumPyro validate its own distribution parameters and raise its existing errors for unsupported enumeration. PyLIMID must validate the additional declarations and mappings it owns. Do not catch every JAX tracing error and reinterpret it as a domain problem: the same exception can come from unrelated code inside a user's callable. NumPyro parameter validation by itself cannot catch a mismatch with PyLIMID labels or policy domains, which NumPyro does not know about.

## Posterior summaries

Continue returning raw JAX draws with joint sample order preserved. Add only the metadata needed to distinguish mathematical discreteness, numerical semantics, and finite domains; avoid storing conflicting copies of the same information.

For categorical variables, retain per-category probabilities and reject numerical summaries of arbitrary category codes. For numerical counts, allow empirical means, standard deviations, and intervals as for other numerical draws. Document that an interval for integer counts may include gaps and is not necessarily a smallest credible set.

An unbounded count cannot have a fixed-length probability vector covering every possible value. Event probabilities such as `P(arrivals >= 20)` can be estimated directly from the draws. The existing `marginal()` contract should stay finite-domain-specific, or any replacement must state which values accompany its returned masses and how omitted outcomes are represented. Never report a vector over observed counts as if unseen counts were impossible.

Observations should use category codes for categorical variables and actual numeric values for counts. Display labels should not silently change evidence encoding. Determining the API for explicit count domains and result metadata is required before implementing this change.

## Parent-dependent trial counts and future inference

The first stage does not support every mathematically well-defined posterior query. It preserves forward simulation and observed-count likelihoods, while rejecting every relevant latent numerical count before either posterior path, including a fixed-bound Binomial. Support for finite numerical enumeration remains a separate extension.

A later investigation could use a fixed global count range for enumeration and assign zero probability to outcomes above the conditional trial count. This needs correct probability masking, normalization, gradients, and handling of NumPyro enumeration dimensions. An adapter that represents those probabilities as a Categorical distribution must preserve numerical count semantics. It should be tested against exact small-model calculations before being exposed.

Unbounded discrete latent inference is a separate feature. NumPyro has additional inference approaches, but their mere availability is not an integration plan. Do not promise that adding a different kernel supports every count distribution, automatically truncate unbounded support, or silently substitute a posterior method.

## Implementation scope and sequence

This affects more than `Posterior`. The current assumptions are spread across chance-node classification, graph helpers, inference dispatch, model translation, policy encoding, backward-induction grouping, both scan implementations, and policy rendering. Structural graph algorithms and decision action representation should mostly retain their current roles.

Categorical validation, classification separation, forward numerical counts and observed-count likelihoods are complete. Current tests cover parent-dependent Binomial trial counts through every simulation path, numerical summaries and joint utility draws, Gamma-Poisson and categorical-parent likelihoods, latent-count rejection with and without continuous latents, unrelated-node pruning, and policy-domain rejection. The remaining sequence is:

1. Settle categorical versus numerical semantics and finite-domain declarations, including how incomplete nodes and result metadata work. Keep the API small; avoid a distribution registry, general type hierarchy, or new public inference switches without a production use case.
2. Introduce a shared domain interpretation and migrate policy construction and rendering. Keep explicit finite domains wherever compilation or table allocation requires them.
3. Preserve forward count sampling and observed-count likelihoods as the domain API evolves. Keep operation-specific checks so a numerical discrete latent is never treated as a continuous NUTS variable.
4. Verify any newly supported finite-discrete posterior queries end to end. Unsupported inference paths should continue to fail explicitly.
5. Investigate bounded parent-dependent enumeration and unbounded latent inference as separate extensions, each with its own correctness criteria.

The library is experimental, so a coherent breaking change can be preferable to parallel compatibility paths. Migration must still be explicit: existing labeled nodes preserve categorical interpretation, count examples can adopt numerical semantics, and continuous nodes and array-valued priors must retain their behavior. A constructor or metadata sketch should be agreed before implementation rather than embedded in this proposal as an established API.

## Acceptance criteria

These criteria cover both implemented behavior and future extensions. Numerical information-set domains, independently editable labels and latent-count inference are not yet implemented.

- Existing weather-based models retain category probabilities, policy indices, readable labels, and results within the existing sampling tolerances.
- A fixed-count Binomial variable can be modeled without stringifying every count, and its empirical mean has count semantics.
- A Poisson variable can feed a numeric utility in forward simulation without a finite list of labels. A suitable count model has an analytical expected utility for a regression reference.
- Every numerical discrete latent, bounded or unbounded, is rejected by the current posterior paths instead of being routed to continuous NUTS or categorical enumeration. An observed Poisson likelihood with continuous latent parameters is supported.
- A count used as a decision parent requires a fixed finite domain. Out-of-domain values cannot be clamped into another policy entry.
- Explicit domains cover all supported parent configurations; a pilot sample is never treated as proof of the domain or of impossibility.
- Adding or changing display labels does not change numerical summaries, evidence values, policy meaning, or sampler selection.
- Forward simulation of the five-or-ten-item batch model succeeds, while unsupported posterior enumeration fails. Any future fixed-domain enumeration workaround agrees with an exact small-model posterior.
- Shared domain mappings work consistently in backward induction, batched scan, intervention scan, contingent-policy inference, and policy rendering. Domain sizes and mappings are not retained across invalidating diagram edits.
- Dirichlet and other array-valued continuous posteriors retain their draw shapes, component summaries, dtypes, and joint sample alignment.

## Recommendation

Separate category labels, numerical meaning, and finite computational domains. Start with scalar counts, retain finite tabular policies, and make operation-specific inference limits explicit. This increases useful expressiveness without committing to a general discrete-inference engine or treating every NumPyro distribution as supported by every PyLIMID operation.
