"""Trainable linear softmax family for categorical decisions."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy._information import encode_features
from pylimid.policy.family import DecisionInformation, PolicyFamily, PolicyParameters


@dataclass(frozen=True)
class LinearSoftmax(PolicyFamily):
    """
    Linear scores over flattened numerical and one-hot categorical information.

    A learned bias supplies an intercept. No scaling is fitted automatically. The
    optimizer fits weights and bias and returns a `CategoricalPolicy`; creators do
    not supply initial parameter values. A decision without parents fits only bias.
    """

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int | None,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize zero weights and bias, giving uniform action probabilities.

        Args:
            rng_key: JAX random key; this family initializes deterministically.
            num_actions: Number of available actions.
            observations: Sample information-parent values in declared parent order.
            parent_states: Categorical parent labels, or ``None`` for numerical inputs.

        Returns:
            A dictionary with ``weights`` and ``bias`` arrays.

        Raises:
            ValueError: If the decision has a continuous action space.
        """
        if num_actions is None:
            raise ValueError("LinearSoftmax requires categorical action states.")
        feature_count = encode_features(observations, parent_states).size
        return {
            "weights": jnp.zeros((feature_count, num_actions)),
            "bias": jnp.zeros(num_actions),
        }

    def distribution(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> dist.CategoricalLogits:
        """
        Construct categorical action probabilities from linear scores.

        Args:
            parameters: The current ``weights`` and ``bias`` arrays.
            observations: Information-parent values in declared parent order.
            parent_states: Categorical parent labels, or ``None`` for numerical inputs.

        Returns:
            A scalar categorical distribution in the declared action order.
        """
        features = encode_features(observations, parent_states)
        return dist.Categorical(
            logits=features @ parameters["weights"] + parameters["bias"]
        )
