Policy families and rules
=========================

A `PolicyFamily` describes what can be fitted for one decision. Declare it on `DecisionNode.policy_family`; `solve` fits its parameters and returns a `CategoricalPolicy`. The joint `FittedPolicy` maps decision names to these local rules. Every rule receives only its decision's declared information parents, including earlier actions or observations that it explicitly remembers.

The names on this page are exported by ``pylimid.policy`` and directly by ``pylimid``. See :doc:`pylimid` for fitting and evaluation, and the :doc:`solving guide <../user_guide/solving>` for worked examples.

.. currentmodule:: pylimid.policy

Families and fitted rules
-------------------------

`TabularSoftmax` uses independent action scores for every combination of scalar categorical information values. Its parents must declare named finite states. `LinearSoftmax` uses a linear function of flattened numerical values and one-hot categorical values, so it can use numerical, categorical or mixed information. Both choose among the decision's finite action labels. A custom family implements `PolicyFamily.initialize` and `PolicyFamily.logits` with pure, differentiable JAX computations.

`CategoricalPolicy.probabilities` returns the action distribution at one information assignment. `CategoricalPolicy.sample` draws an action with an explicit random key, and `CategoricalPolicy.log_prob` evaluates an action's log probability. Numerical observations retain their fitted shapes. Use ``jax.vmap`` for batches; choosing the largest-probability action produces a different policy.

.. autosummary::
   :toctree: generated
   :nosignatures:

   PolicyFamily
   LinearSoftmax
   TabularSoftmax
   CategoricalPolicy

Policy representations
----------------------

The following aliases distinguish a rule for one decision from a mapping of rules for a whole diagram. They describe supported values; they do not construct or validate a policy.

.. py:data:: ActionTable
   :canonical: pylimid.policy.rules.ActionTable
   :type: dict[tuple[int, ...], int]

   A complete deterministic rule for a decision with categorical information parents. Keys contain parent-state indices in `DecisionNode.parents` order; values are action indices in the decision's ``states``, rather than labels. Every combination of parent states must have an entry. A decision without parents uses ``()`` as its only key.

.. py:data:: DecisionRule
   :canonical: pylimid.policy.rules.DecisionRule
   :type: ActionTable | collections.abc.Callable[..., int | jax.Array] | CategoricalPolicy

   A contingent rule for one decision: a deterministic `ActionTable`, a deterministic function or a fitted stochastic `CategoricalPolicy`. A function accepts its information parents by keyword and returns a scalar integer action index as a Python ``int`` or JAX array. It must be JAX-traceable. Functions and fitted rules support forward evaluation without chance-node evidence.

.. py:data:: Policy
   :canonical: pylimid.policy.rules.Policy
   :type: collections.abc.Mapping[str, int | DecisionRule]

   Decision names mapped to fixed action indices or contingent `DecisionRule` values, as accepted by `infer`. A fixed integer action ignores the decision's information parents. Different decisions may use different rule representations in the same mapping. Every decision relevant to the query or evidence must be supplied.

.. py:data:: FittedPolicy
   :canonical: pylimid.policy.rules.FittedPolicy
   :type: dict[str, CategoricalPolicy]

   The joint policy returned by `solve`, containing a fitted stochastic rule for every decision. Each rule retains its family, parameters, information metadata and action labels. Pass the mapping directly to ``infer(policy=solution.policy)`` for forward evaluation; its actions remain stochastic.

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
