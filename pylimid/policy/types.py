"""Shared representations of individual decision rules and joint policies."""

from collections.abc import Callable, Mapping

import jax

from pylimid.policy._categorical.policy import CategoricalPolicy
from pylimid.policy._continuous.policy import ContinuousPolicy

__all__ = ["ActionTable", "DecisionRule", "FittedPolicy", "Policy"]


#: A complete deterministic rule for a decision with categorical information parents.
#: Keys contain parent-state indices in `DecisionNode.parents` order; values are
#: action indices into the decision's ``states``, rather than state labels. Every
#: information assignment must have an entry. A parentless decision uses ``()``
#: as its only key.
type ActionTable = dict[tuple[int, ...], int]


#: A contingent rule for one decision: an `ActionTable`, a deterministic function,
#: or a fitted stochastic `CategoricalPolicy` or `ContinuousPolicy`. Functions
#: receive declared information parents by keyword and return a scalar integer index
#: for categorical actions or a real scalar for continuous actions. Functions must
#: be JAX-traceable and support forward evaluation only. Fitted rules also support
#: posterior conditioning on chance-node evidence.
type DecisionRule = (
    ActionTable
    | Callable[..., int | float | jax.Array]
    | CategoricalPolicy
    | ContinuousPolicy
)

#: Decision names mapped to fixed numerical actions or contingent `DecisionRule`
#: values. A fixed action ignores the decision's information parents. Different
#: decisions may use different representations in the same policy. Every decision
#: relevant to the query or evidence must be supplied when calling `infer`.
type Policy = Mapping[str, int | float | jax.Array | DecisionRule]

#: The joint policy returned by `solve`: decision names mapped exclusively to fitted
#: stochastic `CategoricalPolicy` or `ContinuousPolicy` rules. Each rule retains its
#: family, parameters, information metadata, and action domain. Pass it to `infer`
#: for forward evaluation or conditioning on chance-node evidence. Action
#: probabilities and densities contribute to the joint model during inference.
type FittedPolicy = dict[str, CategoricalPolicy | ContinuousPolicy]
