"""Tests for :class:`decisionpy.inference.exact.linear_gaussian.GaussianFactor`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from decisionpy.inference.exact.linear_gaussian import GaussianFactor


def _moments(factor: GaussianFactor) -> tuple[float, float]:
    """Mean and variance of a single-variable factor."""
    assert len(factor.variables) == 1
    return float(factor.info[0] / factor.precision[0, 0]), float(
        1.0 / factor.precision[0, 0]
    )


# -- from_cpd ---------------------------------------------------------------


def test_from_cpd_root():
    factor = GaussianFactor.from_cpd("x", [], intercept=2.0, slopes=[], scale=3.0)
    assert factor.variables == ("x",)
    assert jnp.allclose(factor.precision, jnp.array([[1.0 / 9.0]]))
    assert jnp.allclose(factor.info, jnp.array([2.0 / 9.0]))


def test_from_cpd_one_parent():
    factor = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    assert factor.variables == ("x1", "x2")
    expected_precision = jnp.array([[9.0, -6.0], [-6.0, 4.0]])
    expected_info = jnp.array([-12.0, 8.0])
    assert jnp.allclose(factor.precision, expected_precision)
    assert jnp.allclose(factor.info, expected_info)


def test_from_cpd_two_parents():
    factor = GaussianFactor.from_cpd(
        "y", ["a", "b"], intercept=1.0, slopes=[2.0, -3.0], scale=2.0
    )
    variance = 4.0
    expected_precision = jnp.array(
        [
            [4.0 / 4.0, -6.0 / 4.0, -2.0 / 4.0],
            [-6.0 / 4.0, 9.0 / 4.0, 3.0 / 4.0],
            [-2.0 / 4.0, 3.0 / 4.0, 1.0 / 4.0],
        ]
    )
    expected_info = jnp.array([-2.0 / 4.0, 3.0 / 4.0, 1.0 / 4.0])
    assert jnp.allclose(factor.precision, expected_precision)
    assert jnp.allclose(factor.info, expected_info)


# -- multiply ---------------------------------------------------------------


def test_multiply_shared_variable_joint():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    joint = f_x1 * f_x2
    assert joint.variables == ("x1", "x2")
    expected_precision = jnp.array([[10.0, -6.0], [-6.0, 4.0]])
    expected_info = jnp.array([-12.0, 8.0])
    assert jnp.allclose(joint.precision, expected_precision)
    assert jnp.allclose(joint.info, expected_info)


def test_multiply_disjoint_variables():
    f_a = GaussianFactor.from_cpd("a", [], intercept=1.0, slopes=[], scale=1.0)
    f_b = GaussianFactor.from_cpd("b", [], intercept=2.0, slopes=[], scale=2.0)
    joint = f_a * f_b
    assert joint.variables == ("a", "b")
    assert jnp.allclose(joint.precision, jnp.diag(jnp.array([1.0, 0.25])))
    assert jnp.allclose(joint.info, jnp.array([1.0, 0.5]))


def test_multiply_commutes():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    ab = f_x1 * f_x2
    ba = f_x2 * f_x1
    assert ab.variables == ("x1", "x2")
    assert ba.variables == ("x1", "x2")
    assert jnp.allclose(ab.precision, ba.precision)
    assert jnp.allclose(ab.info, ba.info)


# -- marginal (Schur complement) --------------------------------------------


def test_marginal_prior_of_child():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    joint = f_x1 * f_x2
    child = joint.marginal(["x2"])
    mean, variance = _moments(child)
    assert jnp.allclose(mean, 2.0)
    assert jnp.allclose(variance, 2.5)


def test_marginal_prior_of_parent():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    joint = f_x1 * f_x2
    parent = joint.marginal(["x1"])
    mean, variance = _moments(parent)
    assert jnp.allclose(mean, 0.0)
    assert jnp.allclose(variance, 1.0)


def test_marginal_keeps_scope_order():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    joint = f_x1 * f_x2
    kept = joint.marginal(["x2", "x1"])
    assert kept.variables == ("x1", "x2")


# -- condition --------------------------------------------------------------


def test_condition_posterior():
    f_x1 = GaussianFactor.from_cpd("x1", [], intercept=0.0, slopes=[], scale=1.0)
    f_x2 = GaussianFactor.from_cpd(
        "x2", ["x1"], intercept=2.0, slopes=[1.5], scale=0.5
    )
    joint = f_x1 * f_x2
    conditioned = joint.condition({"x2": 0.4})
    assert conditioned.variables == ("x1",)
    posterior = conditioned.marginal(["x1"])
    mean, variance = _moments(posterior)
    assert np.isclose(mean, -0.96)
    assert np.isclose(variance, 0.1)


def test_condition_multiple_variables():
    f_a = GaussianFactor.from_cpd("a", [], intercept=0.0, slopes=[], scale=1.0)
    f_b = GaussianFactor.from_cpd("b", [], intercept=0.0, slopes=[], scale=1.0)
    f_y = GaussianFactor.from_cpd(
        "y", ["a", "b"], intercept=0.0, slopes=[1.0, 1.0], scale=1.0
    )
    joint = f_a * f_b * f_y
    conditioned = joint.condition({"a": 1.0, "b": -1.0})
    assert conditioned.variables == ("y",)
    mean, variance = _moments(conditioned)
    assert np.isclose(mean, 0.0)
    assert np.isclose(variance, 1.0)


def test_condition_no_overlap_returns_self():
    f_x = GaussianFactor.from_cpd("x", [], intercept=1.0, slopes=[], scale=1.0)
    conditioned = f_x.condition({"other": 5.0})
    assert conditioned is f_x
