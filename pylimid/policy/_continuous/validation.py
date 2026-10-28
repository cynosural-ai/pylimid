"""Validate scalar continuous policy distributions against declared action domains."""

from typing import SupportsFloat, cast

import numpyro.distributions as dist
from jax.core import Tracer
from numpyro.distributions import constraints

from pylimid.graph import Interval, Real


def validate_distribution(
    distribution: dist.Distribution, action_space: Real | Interval
) -> None:
    """
    Check scalar density support and fixed bounds for a continuous decision.

    Args:
        distribution: Action distribution at one information assignment.
        action_space: Declared scalar continuous domain.

    Raises:
        ValueError: If the distribution is discrete, batched, vector-valued, a point
            mass, has parameter-dependent bounds, or can leave the declared domain.
        NotImplementedError: If its support metadata cannot be classified.
    """
    # A family describes one scalar action at one information assignment;
    # callers add sample and information batch dimensions.
    if distribution.batch_shape or distribution.event_shape:
        raise ValueError(
            "ContinuousPolicy requires a scalar distribution with no batch dimensions."
        )
    support = cast(constraints.Constraint, distribution.support)
    # Point masses do not provide the continuous density this rule scores.
    if isinstance(distribution, dist.Delta) or support.is_discrete:
        raise ValueError("ContinuousPolicy requires a continuous action density.")
    if isinstance(support, constraints.interval):
        bounds = (support.lower_bound, support.upper_bound)
    elif isinstance(support, constraints.greater_than):
        bounds = (support.lower_bound, float("inf"))
    elif isinstance(support, constraints.less_than):
        bounds = (float("-inf"), support.upper_bound)
    elif isinstance(support, type(constraints.real)):
        bounds = (float("-inf"), float("inf"))
    else:
        raise NotImplementedError(
            "Continuous policy support must be real, an interval, or a half-line."
        )
    # The score-function estimator assumes support independent of learned parameters.
    # Domain comparison also needs bounds known during JAX tracing.
    if any(isinstance(bound, Tracer) for bound in bounds):
        raise ValueError(
            "Continuous policy support bounds must be fixed, independent of "
            "trainable parameters and information."
        )
    low, high = (float(cast(SupportsFloat, bound)) for bound in bounds)
    # Checking the whole support also rules out rare invalid actions a sample may miss.
    if isinstance(action_space, Interval) and (
        low < action_space.low or high > action_space.high
    ):
        raise ValueError(
            f"Policy distribution support [{low}, {high}] must stay within "
            f"action space [{action_space.low}, {action_space.high}]."
        )
