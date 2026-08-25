"""Tests for :func:`decisionpy.inference.exact.linear_gaussian.gaussian_cpd`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph import ChanceNode
from decisionpy.inference.exact.linear_gaussian import gaussian_cpd

# --- helpers ----------------------------------------------------------------


def _node(parents=(), fn=None):
    return ChanceNode(name="x", parents=parents, dist=fn)


# -- happy path --------------------------------------------------------------


def test_root_node():
    node = _node(fn=lambda: dist.Normal(2.0, 3.0))
    assert gaussian_cpd(node) == (2.0, [], 3.0)


def test_one_parent():
    node = _node(parents=("p",), fn=lambda p: dist.Normal(2.0 + 1.5 * p, 0.5))
    intercept, slopes, scale = gaussian_cpd(node)
    assert intercept == 2.0
    assert slopes == [1.5]
    assert scale == 0.5


def test_two_parents():
    node = _node(
        parents=("a", "b"),
        fn=lambda a, b: dist.Normal(1.0 + 2.0 * a - 3.0 * b, 0.25),
    )
    intercept, slopes, scale = gaussian_cpd(node)
    assert intercept == 1.0
    assert slopes == [2.0, -3.0]
    assert scale == 0.25


def test_zero_intercept_with_negative_slope():
    node = _node(parents=("p",), fn=lambda p: dist.Normal(-2.0 * p, 1.0))
    intercept, slopes, scale = gaussian_cpd(node)
    assert intercept == 0.0
    assert slopes == [-2.0]
    assert scale == 1.0


def test_jax_scalar_parameters():
    node = _node(
        parents=("p",),
        fn=lambda p: dist.Normal(jnp.asarray(1.5) + 0.5 * p, jnp.asarray(2.0)),
    )
    intercept, slopes, scale = gaussian_cpd(node)
    assert intercept == 1.5
    assert slopes == [0.5]
    assert scale == 2.0


# -- loud failures -----------------------------------------------------------


def test_non_normal_raises():
    node = _node(fn=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])))
    with pytest.raises(TypeError, match="not a Normal"):
        gaussian_cpd(node)


def test_nonlinear_loc_raises():
    node = _node(parents=("p",), fn=lambda p: dist.Normal(2.0 + p**2, 0.5))
    with pytest.raises(ValueError, match="not linear"):
        gaussian_cpd(node)


def test_varying_scale_raises():
    node = _node(
        parents=("p",),
        fn=lambda p: dist.Normal(2.0 + p, 0.5 + 0.1 * p),
    )
    with pytest.raises(ValueError, match="constant scale"):
        gaussian_cpd(node)
