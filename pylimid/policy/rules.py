"""Fitted categorical decision rules and joint-policy representations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy.families import DecisionInformation, PolicyFamily, PolicyParameters

__all__ = ["ActionTable", "CategoricalPolicy", "DecisionRule", "FittedPolicy", "Policy"]


#: A complete deterministic rule for a decision with categorical information parents.
#: Keys contain parent-state indices in `DecisionNode.parents` order; values are
#: action indices into the decision's ``states``, rather than state labels. Every
#: information assignment must have an entry. A parentless decision uses ``()``
#: as its only key.
type ActionTable = dict[tuple[int, ...], int]


#: A contingent rule for one decision: an `ActionTable`, a deterministic function,
#: or a fitted stochastic `CategoricalPolicy`. Functions receive the decision's
#: declared information parents by keyword and return a scalar integer action index
#: as a Python ``int`` or JAX array. Functions must be JAX-traceable and support
#: forward evaluation only. Fitted rules support forward evaluation and posterior
#: conditioning on chance-node evidence.
type DecisionRule = ActionTable | Callable[..., int | jax.Array] | CategoricalPolicy

#: Decision names mapped to fixed action indices or contingent `DecisionRule` values.
#: A fixed integer action ignores the decision's information parents. Different
#: decisions may use different representations in the same policy. Every decision
#: relevant to the query or evidence must be supplied when calling `infer`.
type Policy = Mapping[str, int | DecisionRule]

#: The joint policy returned by `solve`: decision names mapped exclusively to fitted
#: stochastic `CategoricalPolicy` rules. Each rule retains its family, parameters,
#: information metadata, and action labels. Pass this mapping directly to `infer`
#: for forward evaluation or conditioning on chance-node evidence. Action
#: probabilities contribute to the joint model during posterior inference.
type FittedPolicy = dict[str, CategoricalPolicy]


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
        family: The creator-declared `PolicyFamily` that computes action logits.
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
    ) -> dist.CategoricalLogits:
        """Bind observations in declared order and construct the action distribution."""
        unexpected = observations.keys() - self.parents
        if unexpected:
            raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")
        for parent, shape in zip(self.parents, self.parent_shapes, strict=True):
            if jnp.shape(observations[parent]) != shape:
                raise ValueError(
                    f"Information parent {parent!r} must have shape {shape}."
                )
        ordered = {parent: observations[parent] for parent in self.parents}
        parent_states = dict(zip(self.parents, self.parent_states, strict=True))
        logits = self.family.logits(self.parameters, ordered, parent_states)
        if logits.shape != (len(self.states),):
            raise ValueError(
                f"Policy logits must have shape {(len(self.states),)}, "
                f"got {logits.shape}."
            )
        return dist.Categorical(logits=logits)
