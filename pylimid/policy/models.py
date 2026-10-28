"""Models that map local decision information to raw distribution parameters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pylimid.policy.families import (
    DecisionInformation,
    PolicyParameters,
    _features,
)

__all__ = ["Linear", "ParameterModel"]


@dataclass(frozen=True)
class ParameterModel(ABC):
    """
    A trainable map from one information assignment to raw scalar parameters.

    `DistributionPolicy` resolves the output names and transforms raw predictions
    to satisfy the distribution's parameter constraints. Implementations must be
    pure, support JAX transformations, and retain no mutable network state.

    Attributes:
        outputs: Names of the distribution parameters to learn, in prediction order.
            ``None`` asks `DistributionPolicy` to use every parameter not fixed by
            the creator. An empty tuple learns no parameters.
    """

    outputs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate explicitly supplied output names."""
        if self.outputs is None:
            return
        if not isinstance(self.outputs, tuple):
            raise TypeError("`outputs` must be a tuple or None.")
        if any(not isinstance(name, str) or not name.strip() for name in self.outputs):
            raise ValueError("Output names must be non-empty strings.")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Output names must be unique.")

    @abstractmethod
    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_outputs: int,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize the model's trainable parameter tree.

        Args:
            rng_key: JAX random key for initialization.
            num_outputs: Number of scalar outputs resolved by `DistributionPolicy`.
            observations: One information assignment, used to determine input shapes.
            parent_states: Categorical labels, or ``None`` for numerical information.

        Returns:
            A dictionary containing a JAX pytree of trainable floating-point arrays.
        """

    @abstractmethod
    def predict(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """
        Predict raw distribution parameters at one information assignment.

        Args:
            parameters: The model's trainable parameter tree.
            observations: Information values in declared parent order.
            parent_states: Categorical labels, or ``None`` for numerical information.

        Returns:
            A vector with shape ``(num_outputs,)``, in resolved output order.
            `DistributionPolicy` applies parameter-constraint transformations.
        """


@dataclass(frozen=True)
class Linear(ParameterModel):
    """
    Linear raw parameter predictions with a learned intercept for each output.

    Numerical information is flattened and categorical information is one-hot
    encoded, as in `LinearSoftmax`. No scaling or interaction features are added.
    A decision without parents learns only intercepts. Zero initialization gives
    raw outputs of zero, before `DistributionPolicy` applies constraint transforms.

    Attributes:
        outputs: Distribution parameter names, or ``None`` for automatic discovery.
    """

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_outputs: int,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize zero weights and intercepts.

        Args:
            rng_key: JAX random key; this model initializes deterministically.
            num_outputs: Number of raw scalar parameter predictions.
            observations: One information assignment used to determine input shapes.
            parent_states: Categorical labels, or ``None`` for numerical inputs.

        Returns:
            A dictionary with ``weights`` and ``bias`` arrays.
        """
        return {
            "weights": jnp.zeros(
                (_features(observations, parent_states).size, num_outputs)
            ),
            "bias": jnp.zeros(num_outputs),
        }

    def predict(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> jax.Array:
        """
        Calculate linear raw predictions in resolved output order.

        Args:
            parameters: The current ``weights`` and ``bias`` arrays.
            observations: Information values in declared parent order.
            parent_states: Categorical labels, or ``None`` for numerical inputs.

        Returns:
            One raw scalar per distribution parameter to learn.
        """
        return (
            _features(observations, parent_states) @ parameters["weights"]
            + parameters["bias"]
        )
