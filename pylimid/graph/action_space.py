"""Explicit domains for scalar continuous decision actions."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real as RealNumber

__all__ = ["Interval", "Real"]


@dataclass(frozen=True)
class Real:
    """
    An unrestricted real-valued scalar action space.

    Declare this domain on `DecisionNode.action_space`. It specifies the allowed
    values, independently of the policy family used to choose an action.
    """


@dataclass(frozen=True)
class Interval:
    """
    A real-valued scalar action space with fixed, inclusive finite bounds.

    Declare this domain on `DecisionNode.action_space`. Bounds describe allowed
    actions and do not depend on the decision's information parents.

    Attributes:
        low: Inclusive lower bound.
        high: Inclusive upper bound, strictly greater than ``low``.

    Raises:
        TypeError: If a bound is not a real number or is a boolean.
        ValueError: If a bound is not finite, or ``low`` is not less than ``high``.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the fixed scalar bounds."""
        for name, bound in (("low", self.low), ("high", self.high)):
            if isinstance(bound, bool) or not isinstance(bound, RealNumber):
                raise TypeError(f"Interval {name} must be a real number.")
            if not isfinite(bound):
                raise ValueError(f"Interval {name} must be finite.")
        if self.low >= self.high:
            raise ValueError("Interval requires low < high.")
