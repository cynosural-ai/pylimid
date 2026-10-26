"""Trainable policy families that initialize parameters and compute action logits."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "DecisionInformation",
    "PolicyParameters",
    "PolicyFamily",
    "LinearSoftmax",
    "TabularSoftmax",
]

#: Values available to one decision at one information assignment, keyed by its
#: declared parent names and ordered as `DecisionNode.parents`. Categorical values
#: are integer state indices; numerical values retain their scalar or array shapes.
#: There is no leading sample axis. Use ``jax.vmap`` to evaluate several assignments.
type DecisionInformation = Mapping[str, jax.Array]

#: The trainable parameters of one decision's `PolicyFamily`, rooted in a dictionary.
#: Values may be nested dictionaries, sequences, or other JAX pytree containers;
#: leaves are trainable floating-point JAX arrays. Each family owns the keys and
#: shapes. For example, `LinearSoftmax` uses ``weights`` and ``bias``, while
#: `TabularSoftmax` uses ``logits``. A joint optimizer parameter tree maps decision
#: names to these per-decision dictionaries.
type PolicyParameters = dict[str, Any]


class PolicyFamily(ABC):
    """
    A trainable family that initializes parameters and produces action logits.

    Declare a family on `DecisionNode.policy_family`. The optimizer fits its JAX
    parameter tree and returns a `CategoricalPolicy`, which owns normalization,
    sampling, and action log probabilities. Family methods must be pure and support
    JAX compilation; logits must also support differentiation with respect to all
    parameters. They operate on one information assignment at a time and retain no
    mutable state between calls.

    Observation mappings follow the decision's declared parent order. Categorical
    parents supply state indices and named labels; numerical parents have no labels.
    """

    @abstractmethod
    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize the family's trainable parameter tree.

        Args:
            rng_key: JAX random key for parameter initialization.
            num_actions: Number of actions in the decision's declared state order.
            observations: One sampled information assignment in declared parent order,
                used to determine input shapes. Numerical values retain their array
                shapes; categorical values are integer state indices.
            parent_states: Categorical labels by parent name, or ``None`` for
                numerical information.

        Returns:
            A dictionary containing a JAX pytree of trainable floating-point arrays.
            Nested dictionaries and sequences are supported. Use ``rng_key`` for
            random initialization when the family requires it.
        """

    @abstractmethod
    def logits(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """
        Compute unnormalized action scores for one information assignment.

        Args:
            parameters: The family's current trainable parameter tree.
            observations: Information-parent values in declared parent order.
            parent_states: Categorical labels by parent name, or ``None`` for
                numerical information.

        Returns:
            Unnormalized scores with shape ``(num_actions,)``, in the declared action
            order. `CategoricalPolicy` applies softmax to obtain probabilities.
        """


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
        num_actions: int,
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
        """
        feature_count = _features(observations, parent_states).size
        return {
            "weights": jnp.zeros((feature_count, num_actions)),
            "bias": jnp.zeros(num_actions),
        }

    def logits(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """
        Calculate linear action scores from encoded observations.

        Args:
            parameters: The current ``weights`` and ``bias`` arrays.
            observations: Information-parent values in declared parent order.
            parent_states: Categorical parent labels, or ``None`` for numerical inputs.

        Returns:
            One unnormalized score per action.
        """
        features = _features(observations, parent_states)
        return features @ parameters["weights"] + parameters["bias"]


@dataclass(frozen=True)
class TabularSoftmax(PolicyFamily):
    """
    Independent action scores for every scalar categorical information assignment.

    Every parent must declare named finite states and return a scalar state index.
    A parentless decision has one score row. Rows express interactions between
    categorical parents, with no parameter sharing. Their count grows as the product
    of parent cardinalities; rarely visited rows may receive little training.
    """

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize uniform action probabilities for every information assignment.

        Args:
            rng_key: JAX random key; this family initializes deterministically.
            num_actions: Number of available actions.
            observations: Sample scalar parent-state indices in declared parent order.
            parent_states: Named finite states for every information parent.

        Returns:
            A dictionary containing ``logits`` with one leading axis per parent
            and a final action axis.

        Raises:
            ValueError: If a parent lacks named states or has a non-scalar value.
        """
        cards = []
        for parent, value in observations.items():
            labels = parent_states[parent]
            if labels is None:
                raise ValueError(
                    "TabularSoftmax requires named finite states "
                    f"for parent {parent!r}."
                )
            if jnp.shape(value) != ():
                raise ValueError(
                    "TabularSoftmax requires a scalar state index "
                    f"for parent {parent!r}."
                )
            cards.append(len(labels))
        return {"logits": jnp.zeros((*cards, num_actions))}

    def logits(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """
        Look up the scores for a categorical information assignment.

        Args:
            parameters: The current categorical score table under ``logits``.
            observations: Scalar state indices in declared parent order.
            parent_states: Named finite states for every information parent.

        Returns:
            One unnormalized score per action in the selected row.
        """
        return parameters["logits"][tuple(observations.values())]


def _features(
    observations: DecisionInformation,
    parent_states: Mapping[str, tuple[str, ...] | None],
) -> jax.Array:
    """Encode numerical values and named categories in the information-parent order."""
    features = []
    for parent, value in observations.items():
        states = parent_states[parent]
        value = jnp.asarray(value)
        feature = value if states is None else jax.nn.one_hot(value, len(states))
        features.append(jnp.ravel(feature))
    return jnp.concatenate(features) if features else jnp.empty((0,))
