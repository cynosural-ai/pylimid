"""Declared policy families and fitted stochastic decision rules."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

__all__ = ["LinearSoftmax", "SoftmaxPolicy"]


@dataclass(frozen=True)
class LinearSoftmax:
    """
    Linear scores followed by softmax over a decision's finite actions.

    Declare this family through ``DecisionNode.policy_family``. Numerical information
    parents supply their raw values, flattened into features. Categorical parents
    supply one-hot features in their declared state order. A learned bias provides
    an intercept. No scaling is fitted or applied automatically.

    The solver fits weights and biases and returns a `SoftmaxPolicy`. This family
    does not require the creator to supply parameter values.
    """

    def _features(self, parents, parent_states, observations):
        """Encode numerical values and named categories in a fixed feature order."""
        features = []
        for parent, states in zip(parents, parent_states, strict=True):
            value = jnp.asarray(observations[parent])
            feature = value if states is None else jax.nn.one_hot(value, len(states))
            features.append(jnp.ravel(feature))
        return jnp.concatenate(features) if features else jnp.empty((0,))


@dataclass(frozen=True)
class SoftmaxPolicy:
    """
    A fitted linear-softmax rule, including its information and action labels.

    Use `SoftmaxPolicy.probabilities` to inspect the rule and `SoftmaxPolicy.sample`
    to choose an action with an explicit random key. Passing the rule to
    ``infer(policy=...)`` preserves its
    randomized meaning during forward evaluation. Its inputs must have the same
    shapes and categorical state indices used during fitting.

    Attributes:
        family: The creator-selected `LinearSoftmax` declaration.
        parents: Names of the information parents, in feature order.
        parent_states: State labels for each categorical parent; ``None`` for
            numerical parents.
        parent_shapes: Shapes of the information-parent values used during fitting.
        states: Labels for the available actions, in probability order.
        weights: Matrix with shape ``(num_features, num_actions)``.
        bias: Intercepts with shape ``(num_actions,)``.
    """

    family: LinearSoftmax
    parents: tuple[str, ...]
    parent_states: tuple[tuple[str, ...] | None, ...]
    parent_shapes: tuple[tuple[int, ...], ...]
    states: tuple[str, ...]
    weights: jax.Array
    bias: jax.Array

    def _distribution(self, observations: dict[str, jax.Array]):
        unexpected = observations.keys() - self.parents
        if unexpected:
            raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")
        for parent, shape in zip(self.parents, self.parent_shapes, strict=True):
            if jnp.shape(observations[parent]) != shape:
                raise ValueError(
                    f"Information parent {parent!r} must have shape {shape}."
                )
        features = self.family._features(self.parents, self.parent_states, observations)
        return dist.Categorical(logits=features @ self.weights + self.bias)

    def probabilities(self, **observations: jax.Array) -> jax.Array:
        """
        Return action probabilities at the supplied observation.

        Args:
            **observations: Information-parent values keyed by their declared names.

        Returns:
            A probability vector in ``states`` order.
        """
        return self._distribution(observations).probs

    def sample(self, rng_key: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Sample an action index at the supplied observation.

        Args:
            rng_key: JAX random key, supplied positionally.
            **observations: Information-parent values keyed by their declared names.

        Returns:
            A scalar integer index into ``states``.
        """
        return self._distribution(observations).sample(rng_key)

    def log_prob(self, action: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Return an action's log probability at the supplied observation.

        Args:
            action: Action index, supplied positionally.
            **observations: Information-parent values keyed by their declared names.

        Returns:
            The scalar log probability of ``action``.
        """
        return self._distribution(observations).log_prob(action)
