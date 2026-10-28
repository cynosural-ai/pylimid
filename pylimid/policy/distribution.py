"""Compose scalar continuous distributions with trainable parameter models."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import biject_to, constraints
from numpyro.distributions.transforms import Transform

from pylimid.graph import Interval
from pylimid.policy.families import DecisionInformation, PolicyFamily, PolicyParameters
from pylimid.policy.models import ParameterModel

__all__ = ["DistributionPolicy"]


class _DistributionMetadata(Protocol):
    arg_constraints: Mapping[str, constraints.Constraint]
    support: object


@dataclass(frozen=True, init=False)
class DistributionPolicy(PolicyFamily):
    """
    Compose a scalar continuous distribution and a trainable parameter model.

    Output names default to the distribution's ``arg_constraints``, excluding
    ``fixed_parameters``. Explicit `ParameterModel.outputs` may reorder or select
    learned parameters; every remaining parameter must be supplied as fixed.
    NumPyro's constraint bijections transform raw predictions into valid parameters.
    For example, a Normal's location is unchanged and its scale is exponentiated.

    Distribution classes must declare a fixed scalar continuous support and scalar
    parameter constraints, as Normal, Beta, Gamma and LogNormal do. Custom factories
    must expose the same metadata. NumPyro's TruncatedNormal factory is also
    supported with fixed finite ``low`` and ``high`` bounds. Parameter-dependent
    supports and vector parameters require a separate family implementation.

    Args:
        distribution: NumPyro distribution constructor or a compatible factory.
        parameter_model: Model predicting the unconstrained parameters to learn.
        fixed_parameters: Scalar constructor arguments held fixed during training.

    Attributes:
        parameter_model: Model predicting the unconstrained parameters to learn.
        fixed_parameters: Scalar constructor arguments held fixed during training.
            They are copied at construction. Truncation bounds must be fixed.

    Raises:
        TypeError: If the parameter model or fixed scalar parameters are invalid.
        ValueError: If output names and fixed parameters do not cover the constructor
            parameters, fixed values violate their constraints, or truncation bounds
            are missing or invalid.
        NotImplementedError: If the constructor lacks compatible scalar metadata or
            a parameter constraint cannot be transformed.
    """

    _constructor: Callable[..., dist.Distribution] = field(repr=False)
    parameter_model: ParameterModel
    fixed_parameters: Mapping[str, float]
    _parameter_transforms: tuple[tuple[str, Transform], ...] = field(
        init=False, repr=False
    )

    def __init__(
        self,
        distribution: Callable[..., dist.Distribution],
        parameter_model: ParameterModel,
        *,
        fixed_parameters: Mapping[str, float] | None = None,
    ) -> None:
        """
        Resolve and validate a continuous distribution parameterization.

        Args:
            distribution: NumPyro constructor or factory exposing compatible metadata.
            parameter_model: Trainable map from information to raw parameters.
            fixed_parameters: Scalar arguments excluded from optimization.
        """
        if not isinstance(parameter_model, ParameterModel):
            raise TypeError("`parameter_model` must implement ParameterModel.")
        fixed = {} if fixed_parameters is None else dict(fixed_parameters)
        parameter_constraints, fixed_only_parameters = _parameter_schema(
            distribution, fixed
        )
        parameter_transforms = _resolve_outputs(
            parameter_model.outputs, parameter_constraints, fixed, fixed_only_parameters
        )
        _validate_fixed_parameters(fixed, parameter_constraints)

        object.__setattr__(self, "_constructor", distribution)
        object.__setattr__(self, "parameter_model", parameter_model)
        object.__setattr__(self, "fixed_parameters", MappingProxyType(fixed))
        object.__setattr__(self, "_parameter_transforms", parameter_transforms)

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int | None,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize the model for the resolved continuous distribution parameters.

        Args:
            rng_key: JAX random initialization key.
            num_actions: Must be ``None`` for a continuous decision.
            observations: One information assignment used to determine input shapes.
            parent_states: Categorical labels, or ``None`` for numerical information.

        Returns:
            The parameter model's trainable JAX parameter dictionary.

        Raises:
            ValueError: If this family is attached to a categorical decision.
        """
        if num_actions is not None:
            raise ValueError(
                "DistributionPolicy requires a continuous decision action space."
            )
        return self.parameter_model.initialize(
            rng_key,
            num_outputs=len(self._parameter_transforms),
            observations=observations,
            parent_states=parent_states,
        )

    def distribution(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> dist.Distribution:
        """
        Construct an action distribution from transformed model predictions.

        Args:
            parameters: The parameter model's fitted or current parameter tree.
            observations: One information assignment in declared parent order.
            parent_states: Categorical labels, or ``None`` for numerical information.

        Returns:
            A scalar continuous NumPyro distribution. The action domain is checked
            when the distribution is bound by `ContinuousPolicy`.

        Raises:
            ValueError: If the parameter model returns an incompatible output shape.
        """
        raw = self.parameter_model.predict(parameters, observations, parent_states)
        # The output axis names parameters, rather than actions or sample batches.
        expected_shape = (len(self._parameter_transforms),)
        if raw.shape != expected_shape:
            raise ValueError(
                f"Parameter model must return shape {expected_shape}, got {raw.shape}."
            )
        values: dict[str, float | jax.Array] = dict(self.fixed_parameters)
        for prediction, (name, transform) in zip(
            raw, self._parameter_transforms, strict=True
        ):
            values[name] = transform(prediction)
        return self._constructor(**values)


def _parameter_schema(
    distribution: Callable[..., dist.Distribution], fixed: Mapping[str, float]
) -> tuple[Mapping[str, constraints.Constraint], set[str]]:
    """Read scalar parameter constraints and fixed-only constructor arguments."""
    if distribution is dist.TruncatedNormal:
        # This NumPyro factory lacks class metadata. Learn the Normal parameters
        # while keeping the truncation bounds fixed.
        if "low" not in fixed or "high" not in fixed:
            raise ValueError("TruncatedNormal requires fixed `low` and `high` bounds.")
        Interval(fixed["low"], fixed["high"])
        return dist.Normal.arg_constraints, {"low", "high"}
    if not hasattr(distribution, "arg_constraints"):
        raise NotImplementedError(
            "Distribution constructors must expose arg_constraints and fixed "
            "scalar support metadata; use a custom PolicyFamily otherwise."
        )
    metadata = cast(_DistributionMetadata, distribution)
    support = metadata.support
    # Instance-dependent support (such as Uniform's) cannot be checked here,
    # even when the caller intends to fix all its bounds.
    if (
        isinstance(support, property)
        or not isinstance(support, constraints.Constraint)
        or support.is_discrete
        or support.event_dim != 0
    ):
        raise NotImplementedError(
            "DistributionPolicy requires fixed scalar continuous support."
        )
    return metadata.arg_constraints, set()


def _resolve_outputs(
    outputs: tuple[str, ...] | None,
    parameter_constraints: Mapping[str, constraints.Constraint],
    fixed_parameters: Mapping[str, float],
    fixed_only_parameters: set[str],
) -> tuple[tuple[str, Transform], ...]:
    """Resolve learned parameter names and constraint transforms in output order."""
    # Truncation bounds are valid constructor arguments without being model outputs.
    unknown = (
        fixed_parameters.keys() - parameter_constraints.keys() - fixed_only_parameters
    )
    if unknown:
        raise ValueError(f"Unknown fixed distribution parameters: {sorted(unknown)}.")
    if outputs is None:
        outputs = tuple(
            name for name in parameter_constraints if name not in fixed_parameters
        )
    # Every distribution parameter needs exactly one source: learned or fixed.
    if set(outputs) & fixed_parameters.keys():
        raise ValueError("A distribution parameter cannot be both learned and fixed.")
    if (
        set(outputs) | (fixed_parameters.keys() - fixed_only_parameters)
        != parameter_constraints.keys()
    ):
        raise ValueError(
            "Learned outputs and fixed parameters must cover exactly "
            f"{tuple(parameter_constraints)}."
        )
    parameter_transforms = []
    for name in outputs:
        constraint = parameter_constraints[name]
        # Each model output is one differentiable scalar, not a vector or category.
        if constraint.is_discrete or constraint.event_dim != 0:
            raise NotImplementedError(
                "Learned distribution parameters must be scalar and continuous."
            )
        parameter_transforms.append((name, biject_to(constraint)))
    return tuple(parameter_transforms)


def _validate_fixed_parameters(
    fixed_parameters: Mapping[str, float],
    parameter_constraints: Mapping[str, constraints.Constraint],
) -> None:
    """Check fixed scalar values for finiteness and distribution constraints."""
    for name, value in fixed_parameters.items():
        array = jnp.asarray(value)
        if array.shape or not (
            jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)
        ):
            raise TypeError(f"Fixed parameter {name!r} must be a real scalar.")
        # Fixed values bypass the learned outputs' constraint transforms.
        if not bool(jnp.isfinite(array)):
            raise ValueError(f"Fixed parameter {name!r} must be finite.")
        if name in parameter_constraints and not bool(
            parameter_constraints[name](array)
        ):
            raise ValueError(
                f"Fixed parameter {name!r} violates its distribution constraint."
            )
