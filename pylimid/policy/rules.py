"""Fitted categorical decision rules and joint-policy representations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy.families import PolicyFamily

__all__ = ["CategoricalPolicy", "Policy"]


@dataclass(frozen=True)
class CategoricalPolicy:
    """
    A fitted categorical rule with parameters, local information, and action labels.

    `CategoricalPolicy.probabilities` inspects the action distribution;
    `CategoricalPolicy.sample` samples an action with an explicit random key. Forward
    evaluation through ``infer(policy=...)`` preserves its stochastic behavior.
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
    parameters: dict[str, Any]

    def _distribution(self, observations: Mapping[str, jax.Array]):
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
        return self._distribution(observations).sample(rng_key)

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


#: Decision names mapped to action tables, deterministic functions, or fitted rules.
#: Tables map parent-state tuples to action indices. Functions accept information
#: parents by keyword and return a scalar integer action index. Functions and fitted
#: stochastic rules currently support forward evaluation only.
Policy: TypeAlias = Mapping[
    str,
    dict[tuple[int, ...], int] | Callable[..., int | jax.Array] | CategoricalPolicy,
]
