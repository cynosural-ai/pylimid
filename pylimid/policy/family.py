"""Shared interfaces and types for trainable policy families."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import jax
import numpyro.distributions as dist

__all__ = [
    "DecisionInformation",
    "PolicyFamily",
    "PolicyParameters",
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
    A trainable family that initializes parameters and constructs action distributions.

    Declare a family on `DecisionNode.policy_family`. The optimizer fits its JAX
    parameter tree and returns a `CategoricalPolicy` or `ContinuousPolicy`, which
    binds the family's distributions to the decision's information and action domain.
    Family methods
    must be pure and support JAX compilation. Distribution log probabilities must
    support differentiation with respect to all trainable parameters. Methods operate
    on one information assignment at a time and retain no mutable state between calls.

    Observation mappings follow the decision's declared parent order. Categorical
    parents supply state indices and named labels; numerical parents have no labels.
    """

    @abstractmethod
    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int | None,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize the family's trainable parameter tree.

        Args:
            rng_key: JAX random key for parameter initialization.
            num_actions: Number of categorical actions, or ``None`` for a continuous
                decision with an explicit action space.
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
    def distribution(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> dist.Distribution:
        """
        Construct the action distribution for one information assignment.

        Args:
            parameters: The family's current trainable parameter tree.
            observations: Information-parent values in declared parent order.
            parent_states: Categorical labels by parent name, or ``None`` for
                numerical information.

        Returns:
            A NumPyro distribution supporting sampling and scalar action log
            probabilities. Categorical decisions require a scalar categorical
            distribution with one probability per action in declared state order.
            Do not add a leading sample or information-assignment axis.
        """
