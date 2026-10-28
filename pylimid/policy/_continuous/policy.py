"""Fitted scalar continuous decision rules."""

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from pylimid.graph import Interval, Real
from pylimid.policy._continuous.validation import validate_distribution
from pylimid.policy._information import bind_information
from pylimid.policy.family import DecisionInformation, PolicyFamily, PolicyParameters


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
        ordered = bind_information(self.parents, self.parent_shapes, observations)
        parent_states = dict(zip(self.parents, self.parent_states, strict=True))
        distribution = self.family.distribution(self.parameters, ordered, parent_states)
        validate_distribution(distribution, self.action_space)
        return distribution
