"""Fitted categorical decision rules and action distribution checks."""

from dataclasses import dataclass
from typing import cast

import jax
import numpyro.distributions as dist

from pylimid.policy._information import bind_information
from pylimid.policy.family import DecisionInformation, PolicyFamily, PolicyParameters


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
        ordered = bind_information(self.parents, self.parent_shapes, observations)
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
