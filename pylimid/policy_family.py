"""Declared policy families and fitted stochastic decision rules."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

__all__ = ["LinearSoftmax", "SoftmaxPolicy", "TabularSoftmax", "TabularPolicy"]


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

    def _initialize(self, parents, parent_states, observations, states):
        """Bind information metadata and initialize uniform action probabilities."""
        feature_count = self._features(parents, parent_states, observations).size
        return SoftmaxPolicy(
            family=self,
            parents=parents,
            parent_states=parent_states,
            parent_shapes=tuple(jnp.shape(observations[parent]) for parent in parents),
            states=states,
            weights=jnp.zeros((feature_count, len(states))),
            bias=jnp.zeros(len(states)),
        )

    def _features(self, parents, parent_states, observations):
        """Encode numerical values and named categories in a fixed feature order."""
        features = []
        for parent, states in zip(parents, parent_states, strict=True):
            value = jnp.asarray(observations[parent])
            feature = value if states is None else jax.nn.one_hot(value, len(states))
            features.append(jnp.ravel(feature))
        return jnp.concatenate(features) if features else jnp.empty((0,))


@dataclass(frozen=True)
class TabularSoftmax:
    """
    Independent softmax action scores for every categorical information assignment.

    Declare this family through ``DecisionNode.policy_family``. Every information
    parent must have named finite states and a scalar state index. Earlier decisions
    can be information parents. A decision without parents has one action-score row.

    The solver fits a separate row for every parent-state combination and returns a
    `TabularPolicy`. Rows can express interactions between categorical parents. Their
    count grows as the product of the parents' state counts; rarely visited rows may
    receive little training. No rows or parameters are shared between assignments.
    """

    def _initialize(self, parents, parent_states, observations, states):
        """Bind scalar categorical information and initialize uniform table rows."""
        cards = []
        for parent, labels in zip(parents, parent_states, strict=True):
            if labels is None:
                raise ValueError(
                    "TabularSoftmax requires named finite states "
                    f"for parent {parent!r}."
                )
            if jnp.shape(observations[parent]) != ():
                raise ValueError(
                    "TabularSoftmax requires a scalar state index "
                    f"for parent {parent!r}."
                )
            cards.append(len(labels))
        return TabularPolicy(
            family=self,
            parents=parents,
            parent_states=parent_states,
            parent_shapes=tuple(() for _ in parents),
            states=states,
            logits=jnp.zeros((*cards, len(states))),
        )


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

    def _parameters(self):
        """Return the fitted arrays updated by the policy-gradient solver."""
        return {"weights": self.weights, "bias": self.bias}

    def _distribution(self, observations: dict[str, jax.Array]):
        _check_observations(self.parents, self.parent_shapes, observations)
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


@dataclass(frozen=True)
class TabularPolicy:
    """
    A fitted stochastic action table with categorical information and action labels.

    Each entry of ``logits`` contains one score per action. Its leading axes follow
    ``parents`` and their state-label order; its last axis follows ``states``.
    Without parents, ``logits`` is a single action-score vector.

    Use `TabularPolicy.probabilities` to inspect a row and `TabularPolicy.sample`
    to sample with an explicit random key. Passing the rule to ``infer(policy=...)``
    preserves its randomized meaning during forward evaluation. This is an
    approximate policy-gradient result, with no optimality guarantee. Taking each
    row's largest score defines a different policy requiring separate evaluation.

    Attributes:
        family: The creator-selected `TabularSoftmax` declaration.
        parents: Names of the categorical information parents, in table-axis order.
        parent_states: State labels for each information parent, in axis order.
        parent_shapes: Scalar shapes ``()`` for every information parent.
        states: Available action labels, in probability order.
        logits: Fitted scores with shape ``(*parent_state_counts, num_actions)``.
    """

    family: TabularSoftmax
    parents: tuple[str, ...]
    parent_states: tuple[tuple[str, ...], ...]
    parent_shapes: tuple[tuple[int, ...], ...]
    states: tuple[str, ...]
    logits: jax.Array

    def _parameters(self):
        """Return the fitted arrays updated by the policy-gradient solver."""
        return {"logits": self.logits}

    def _distribution(self, observations: dict[str, jax.Array]):
        _check_observations(self.parents, self.parent_shapes, observations)
        row = self.logits[tuple(observations[parent] for parent in self.parents)]
        return dist.Categorical(logits=row)

    def probabilities(self, **observations: jax.Array) -> jax.Array:
        """
        Return action probabilities for a categorical information assignment.

        Args:
            **observations: Scalar information-parent state indices, keyed by name.

        Returns:
            A probability vector in ``states`` order.
        """
        return self._distribution(observations).probs

    def sample(self, rng_key: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Sample an action index for a categorical information assignment.

        Args:
            rng_key: JAX random key, supplied positionally.
            **observations: Scalar information-parent state indices, keyed by name.

        Returns:
            A scalar integer index into ``states``.
        """
        return self._distribution(observations).sample(rng_key)

    def log_prob(self, action: jax.Array, /, **observations: jax.Array) -> jax.Array:
        """
        Return an action's log probability for a categorical information assignment.

        Args:
            action: Action index, supplied positionally.
            **observations: Scalar information-parent state indices, keyed by name.

        Returns:
            The scalar log probability of ``action``.
        """
        return self._distribution(observations).log_prob(action)


def _check_observations(parents, parent_shapes, observations):
    """Require the fitted information binding and shapes when evaluating a rule."""
    unexpected = observations.keys() - parents
    if unexpected:
        raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")
    for parent, shape in zip(parents, parent_shapes, strict=True):
        if jnp.shape(observations[parent]) != shape:
            raise ValueError(f"Information parent {parent!r} must have shape {shape}.")
