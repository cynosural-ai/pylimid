Policy families and rules
=========================

A `PolicyFamily` describes what can be fitted for one decision. Declare it on `DecisionNode.policy_family`; `solve` fits its parameters and returns a `CategoricalPolicy` or `ContinuousPolicy`. The joint `FittedPolicy` maps decision names to these local rules. Every rule receives only its decision's declared information parents, including earlier actions or observations that it explicitly remembers.

The names on this page are exported by ``pylimid.policy`` and directly by ``pylimid``. See :doc:`pylimid` for fitting and evaluation, and the :doc:`solving guide <../user_guide/solving>` for worked examples.

.. currentmodule:: pylimid.policy

Families and fitted rules
-------------------------

`TabularSoftmax` uses independent action scores for every combination of scalar categorical information values. Its parents must declare named finite states. `LinearSoftmax` uses a linear function of flattened numerical values and one-hot categorical values, so it can use numerical, categorical or mixed information. Both choose among the decision's finite action labels. A custom family implements `PolicyFamily.initialize` and `PolicyFamily.distribution` with pure JAX computations and differentiable distribution log probabilities.

``distribution`` returns a NumPyro action distribution without batch dimensions. Categorical decisions require a categorical distribution, using probabilities or logits, with one probability per declared action. Continuous decisions require a scalar continuous density with fixed support contained in their action domain. ``initialize`` receives ``num_actions=None`` for continuous decisions. Custom categorical families that previously implemented ``logits`` should return ``dist.Categorical(logits=...)`` from ``distribution`` instead. The declarations ``LinearSoftmax()`` and ``TabularSoftmax()`` are unchanged.

`CategoricalPolicy.probabilities` returns the action distribution at one information assignment. `CategoricalPolicy.sample` draws an action with an explicit random key, and `CategoricalPolicy.log_prob` evaluates an action's log probability. Numerical observations retain their fitted shapes. Use ``jax.vmap`` for batches; choosing the largest-probability action produces a different policy.

`ContinuousPolicy.distribution` returns the conditional NumPyro distribution for inspection. `ContinuousPolicy.sample` draws a scalar action and `ContinuousPolicy.log_prob` evaluates its log density. Both fitted rule types can be reused with `infer`, including conditioning on chance-node evidence. A continuous fitted rule retains its declared `Real` or `Interval` domain.

.. autosummary::
   :toctree: generated
   :nosignatures:

   PolicyFamily
   LinearSoftmax
   TabularSoftmax
   CategoricalPolicy
   ContinuousPolicy

Composing continuous policies
-----------------------------

`DistributionPolicy` combines a NumPyro distribution constructor with a `ParameterModel`. `Linear` predicts raw parameters from flattened numerical and one-hot categorical information. Its ``outputs=None`` default discovers parameter names from the distribution's ``arg_constraints``. NumPyro's constraint bijections transform predictions into valid parameters. For a Normal this leaves ``loc`` unchanged and exponentiates the raw ``scale`` prediction. Zero initialization gives ``loc=0`` and ``scale=1``.

.. code-block:: python

   import numpyro.distributions as dist
   from pylimid import DecisionNode, DistributionPolicy, Linear, Real

   adjustment = DecisionNode(
       "adjustment",
       action_space=Real(),
       policy_family=DistributionPolicy(
           distribution=dist.Normal,
           parameter_model=Linear(),
       ),
   )

Explicit output names select or reorder learned parameters. Every remaining parameter must be supplied through ``fixed_parameters``. To learn location while keeping the action spread fixed:

.. code-block:: python

   family = DistributionPolicy(
       distribution=dist.Normal,
       parameter_model=Linear(outputs=("loc",)),
       fixed_parameters={"scale": 0.2},
   )

An interval does not automatically truncate or scale a distribution. A plain Normal cannot be used inside a bounded action domain. A Beta fits ``Interval(0, 1)``. For another interval, explicitly choose a distribution whose support stays within it, such as a truncated Normal with fixed bounds:

.. code-block:: python

   from pylimid import Interval

   power = DecisionNode(
       "power",
       action_space=Interval(0.0, 100.0),
       policy_family=DistributionPolicy(
           distribution=dist.TruncatedNormal,
           parameter_model=Linear(),
           fixed_parameters={"low": 0.0, "high": 100.0},
       ),
   )

Automatic composition supports constructors with fixed scalar continuous support and scalar parameter constraints, including Normal, Beta, Gamma and LogNormal. TruncatedNormal requires fixed finite bounds. Other factories must expose compatible ``arg_constraints`` and support metadata. Constructors with dependent support metadata, such as Uniform, and vector-valued parameters need a custom `PolicyFamily`. Support must remain independent of trainable parameters and information. Transformed distributions must report accurate support metadata.

Custom parameter models implement `ParameterModel.initialize` and `ParameterModel.predict`, returning a trainable parameter dictionary and one raw scalar per resolved output. Numerical input scaling is explicit. Learning a spread toward zero can make score gradients noisy. A fixed spread restricts the policy family, and neither configuration guarantees convergence or optimality.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistributionPolicy
   ParameterModel
   Linear

Policy representations
----------------------

The following aliases distinguish a rule for one decision from a mapping of rules for a whole diagram. They describe supported values; they do not construct or validate a policy.

.. py:data:: ActionTable
   :canonical: pylimid.policy.rules.ActionTable
   :type: dict[tuple[int, ...], int]

   A complete deterministic rule for a decision with categorical information parents. Keys contain parent-state indices in `DecisionNode.parents` order; values are action indices in the decision's ``states``, rather than labels. Every combination of parent states must have an entry. A decision without parents uses ``()`` as its only key.

.. py:data:: DecisionRule
   :canonical: pylimid.policy.rules.DecisionRule
   :type: ActionTable | collections.abc.Callable[..., int | float | jax.Array] | CategoricalPolicy | ContinuousPolicy

   A contingent rule for one decision: a categorical deterministic `ActionTable`, a deterministic function, or a fitted stochastic `CategoricalPolicy` or `ContinuousPolicy`. A function accepts its information parents by keyword and returns a scalar integer index for categorical actions or a real scalar for continuous actions. It must be JAX-traceable and supports forward evaluation only. Fitted rules also support conditioning on chance-node evidence.

.. py:data:: Policy
   :canonical: pylimid.policy.rules.Policy
   :type: collections.abc.Mapping[str, int | float | jax.Array | DecisionRule]

   Decision names mapped to fixed numerical actions, including scalar JAX arrays, or contingent `DecisionRule` values, as accepted by `infer`. A fixed action ignores the decision's information parents. Fixed actions and deterministic functions must respect their decision's domain. Different decisions may use different rule representations in the same mapping. Every decision relevant to the query or evidence must be supplied.

.. py:data:: FittedPolicy
   :canonical: pylimid.policy.rules.FittedPolicy
   :type: dict[str, CategoricalPolicy | ContinuousPolicy]

   The joint policy returned by `solve`, containing a fitted stochastic rule for every decision. Each rule retains its family, parameters, information metadata and action domain. Pass the mapping directly to ``infer(policy=solution.policy)`` for forward evaluation or combine it with ``observed=`` to condition on chance-node evidence. Its action probabilities and densities contribute to the joint model.

Family inputs and parameters
----------------------------

.. py:data:: DecisionInformation
   :canonical: pylimid.policy.families.DecisionInformation
   :type: collections.abc.Mapping[str, jax.Array]

   Values available to one decision at one information assignment, keyed by its declared parent names in `DecisionNode.parents` order. Categorical values are integer state indices; numerical values retain their scalar or array shapes. There is no leading sample axis. Family methods operate on one assignment at a time; use ``jax.vmap`` to evaluate several assignments.

.. py:data:: PolicyParameters
   :canonical: pylimid.policy.families.PolicyParameters
   :type: dict[str, typing.Any]

   The trainable parameter tree for one decision, rooted in a dictionary. Values may contain nested dictionaries, sequences or other JAX pytree containers whose leaves are trainable floating-point JAX arrays. Each family owns its keys and shapes: `LinearSoftmax` stores ``weights`` and ``bias``, while `TabularSoftmax` stores ``logits``. The optimizer's joint parameter tree maps decision names to these per-decision dictionaries.
