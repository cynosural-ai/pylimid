"""Validation of immutable scalar continuous action domains."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pylimid import Interval


@pytest.mark.parametrize(
    ("low", "high"),
    [(0, 100), (-2.5, 3.5), (np.float32(-1.0), np.float64(2.0))],
)
def test_interval_accepts_fixed_finite_real_bounds(low, high):
    domain = Interval(low, high)
    assert domain.low == low
    assert domain.high == high


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.0, 0.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("nan")),
        (float("-inf"), 1.0),
        (0.0, float("inf")),
    ],
)
def test_interval_rejects_nonfinite_or_unordered_bounds(low, high):
    with pytest.raises(ValueError, match="finite|low < high"):
        Interval(low, high)


@pytest.mark.parametrize("value", [None, "1", True, 1j, [1.0], np.array([1.0])])
@pytest.mark.parametrize("bound", ["low", "high"])
def test_interval_rejects_nonreal_scalar_bounds(value, bound):
    bounds = {"low": 0.0, "high": 2.0, bound: value}
    with pytest.raises(TypeError, match=f"Interval {bound} must be a real number"):
        Interval(**bounds)


def test_interval_bounds_cannot_be_mutated_after_validation():
    domain = Interval(0.0, 100.0)
    with pytest.raises(FrozenInstanceError):
        domain.high = -1.0  # ty: ignore[invalid-assignment]
