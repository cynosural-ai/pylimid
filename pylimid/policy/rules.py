"""Fitted categorical and continuous rules and joint-policy representations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from pylimid.graph import Interval, Real
from pylimid.policy._continuous import validate_distribution
from pylimid.policy.families import DecisionInformation, PolicyFamily, PolicyParameters

__all__ = [
    "ActionTable",
    "CategoricalPolicy",
    "ContinuousPolicy",
    "DecisionRule",
    "FittedPolicy",
    "Policy",
]


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


@dataclass(frozen=True)
class CategoricalPolicy:
    """
    A fitted categorical rule with parameters, local information, and action labels.

    `CategoricalPolicy.probabilities` inspects the action distribution;
    `CategoricalPolicy.sample` samples an action with an explicit random key.
    Evaluation through ``infer(policy=...)`` preserves its stochastic behavior;
    conditioning includes its action probabilities in the joint model.
    Taking the largest probability defines a different policy requiring evaluation.

    Observation methods accept one information assignment by keyword. Categorical
    observations are integer state indices; numerical observations must retain their
    fitted shapes. Use ``jax.vmap`` to evaluate a batch of assignments.

    Attributes:
        family: The creator-declared `PolicyFamily` that constructs action
            distributions.
        parents: Information-parent names, in declared order.
        parent_states: Categorical labels for each parent; ``None`` for numerical
            information.
        parent_shapes: Shapes of the information-parent values used during fitting.
        states: Available action labels, in probability order.
        parameters: The fitted family's JAX parameter tree. `LinearSoftmax` stores
            ``weights`` and ``bias``; `TabularSoftmax` stores ``logits``.
    """

    family: PolicyFamily
    parents: tuple[str, ...]
    parent_states: tuple[tuple[str, ...] | None, ...]
    parent_shapes: tuple[tuple[int, ...], ...]
    states: tuple[str, ...]
    parameters: PolicyParameters

    def probabilities(self, **observations: jax.Array) -> jax.Array:
        """
        Return action probabilities at the supplied observation.

        Args:
            **observations: Declared information-parent values keyed by name.

        Returns:
            A probability vector in ``states`` order.
        """
        return self._distribution(observations).probs

    def sample(self, rng_key: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Sample an action index at the supplied observation.

        Args:
            rng_key: JAX random key, supplied positionally.
            **observations: Declared information-parent values keyed by name.

        Returns:
            A scalar integer index into ``states``.
        """
        # NumPyro's ArrayLike hint is broader than this sampler's JAX array output.
        return cast(jax.Array, self._distribution(observations).sample(rng_key))

    def log_prob(self, action: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Return an action's log probability at the supplied observation.

        Args:
            action: Action index, supplied positionally.
            **observations: Declared information-parent values keyed by name.

        Returns:
            The scalar log probability of ``action``.
        """
        return self._distribution(observations).log_prob(action)

    def _distribution(
        self, observations: DecisionInformation
    ) -> dist.CategoricalLogits | dist.CategoricalProbs:
        """Bind observations and check the categorical action distribution."""
        ordered = _bind_information(self.parents, self.parent_shapes, observations)
        parent_states = dict(zip(self.parents, self.parent_states, strict=True))
        distribution = self.family.distribution(self.parameters, ordered, parent_states)
        if not isinstance(
            distribution, (dist.CategoricalLogits, dist.CategoricalProbs)
        ):
            raise ValueError(
                "CategoricalPolicy requires a Categorical distribution; "
                f"got {type(distribution).__name__}."
            )
        if distribution.batch_shape:
            raise ValueError(
                "CategoricalPolicy requires a scalar distribution with no batch "
                f"dimensions; got batch shape {distribution.batch_shape}."
            )
        probabilities_or_logits = (
            distribution.logits
            if isinstance(distribution, dist.CategoricalLogits)
            else distribution.probs
        )
        num_actions = probabilities_or_logits.shape[-1]
        if num_actions != len(self.states):
            raise ValueError(
                f"Policy distribution must have {len(self.states)} actions, "
                f"got {num_actions}."
            )
        return distribution


@dataclass(frozen=True)
class ContinuousPolicy:
    """
    A fitted stochastic rule for one scalar continuous decision.

    `ContinuousPolicy.distribution` inspects the action distribution at one
    information assignment. `ContinuousPolicy.sample` samples an action, and
    `ContinuousPolicy.log_prob` evaluates its log density. Observations are passed
    by parent name, retaining their fitted shapes. Use ``jax.vmap`` for batches.

    Attributes:
        family: Creator-selected trainable `PolicyFamily`.
        parents: Information-parent names in declared order.
        parent_states: Categorical labels, or ``None`` for numerical information.
        parent_shapes: Shapes of parent values used during fitting.
        action_space: Declared `Real` or `Interval` domain.
        parameters: Fitted trainable parameter dictionary.
    """

    family: PolicyFamily
    parents: tuple[str, ...]
    parent_states: tuple[tuple[str, ...] | None, ...]
    parent_shapes: tuple[tuple[int, ...], ...]
    action_space: Real | Interval
    parameters: PolicyParameters

    def distribution(self, **observations: jax.Array) -> dist.Distribution:
        """
        Return the conditional action distribution.

        Args:
            **observations: Declared information-parent values keyed by name.

        Returns:
            The scalar continuous NumPyro action distribution.

        Raises:
            ValueError: If information shapes or distribution support are incompatible.
            TypeError: If an undeclared information parent is supplied.
            KeyError: If a declared information parent is missing.
            NotImplementedError: If the support metadata is unsupported.
        """
        return self._distribution(observations)

    def sample(self, rng_key: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Sample an action at one information assignment.

        Args:
            rng_key: JAX random key, supplied positionally.
            **observations: Declared information-parent values keyed by name.

        Returns:
            A real-valued scalar in the declared action domain.
        """
        return cast(jax.Array, self._distribution(observations).sample(rng_key))

    def log_prob(self, action: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Evaluate an action's log density at one information assignment.

        Args:
            action: Scalar action, supplied positionally.
            **observations: Declared information-parent values keyed by name.

        Returns:
            The scalar log density, or negative infinity outside distribution support.
        """
        distribution = self._distribution(observations)
        support = cast(constraints.Constraint, distribution.support)
        return jnp.where(support(action), distribution.log_prob(action), -jnp.inf)

    def _distribution(self, observations: DecisionInformation) -> dist.Distribution:
        """Bind local information and validate the continuous action distribution."""
        ordered = _bind_information(self.parents, self.parent_shapes, observations)
        parent_states = dict(zip(self.parents, self.parent_states, strict=True))
        distribution = self.family.distribution(self.parameters, ordered, parent_states)
        validate_distribution(distribution, self.action_space)
        return distribution


def _bind_information(
    parents: tuple[str, ...],
    parent_shapes: tuple[tuple[int, ...], ...],
    observations: DecisionInformation,
) -> dict[str, jax.Array]:
    """Order one information assignment and check its declared parent shapes."""
    unexpected = observations.keys() - parents
    if unexpected:
        raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")
    for parent, shape in zip(parents, parent_shapes, strict=True):
        if jnp.shape(observations[parent]) != shape:
            raise ValueError(f"Information parent {parent!r} must have shape {shape}.")
    return {parent: observations[parent] for parent in parents}
