"""Posterior summaries for scalar and array-valued draws."""

from math import sqrt

import jax
import jax.numpy as jnp
import pytest

from pylimid import Posterior


def test_scalar_summaries_keep_float_results():
    posterior = Posterior(values=[0.0, 2.0, 4.0, 6.0])

    assert posterior.mean() == 3.0
    assert posterior.std() == pytest.approx(sqrt(5))
    assert posterior.hdi(prob=0.75) == (0.0, 4.0)
    assert posterior.hdi(prob=1.0) == (0.0, 6.0)
    assert all(
        isinstance(value, float)
        for value in (posterior.mean(), posterior.std(), *posterior.hdi())
    )


def test_vector_mean_and_standard_deviation_are_componentwise():
    posterior = Posterior(values=jnp.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]]))

    assert isinstance(posterior.mean(), jax.Array)
    assert isinstance(posterior.std(), jax.Array)
    assert posterior.mean() == pytest.approx([3.0, 14.0])
    assert posterior.std() == pytest.approx([sqrt(8 / 3), sqrt(32 / 3)])


def test_vector_hdi_selects_a_separate_interval_for_each_component():
    posterior = Posterior(
        values=jnp.array([[2, 101], [200, 0], [1, 102], [0, 103], [100, 100]])
    )

    lower, upper = posterior.hdi(prob=0.6)

    assert isinstance(lower, jax.Array)
    assert isinstance(upper, jax.Array)
    assert lower == pytest.approx([0, 100])
    assert upper == pytest.approx([2, 102])


@pytest.mark.parametrize(
    ("prob", "expected_lower", "expected_upper"),
    [(1.0, [0, 0], [200, 103]), (0.01, [0, 0], [0, 0])],
)
def test_vector_hdi_extreme_coverage(prob, expected_lower, expected_upper):
    posterior = Posterior(
        values=jnp.array([[2, 101], [200, 0], [1, 102], [0, 103], [100, 100]])
    )

    lower, upper = posterior.hdi(prob=prob)

    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_matrix_summaries_preserve_the_node_shape():
    posterior = Posterior(
        values=jnp.array(
            [
                [[1.0, 10.0], [100.0, 1000.0]],
                [[3.0, 14.0], [104.0, 1006.0]],
                [[5.0, 18.0], [108.0, 1012.0]],
            ]
        )
    )

    assert posterior.mean() == pytest.approx(jnp.array([[3, 14], [104, 1006]]))
    assert posterior.std() == pytest.approx(
        jnp.array([[sqrt(8 / 3), sqrt(32 / 3)], [sqrt(32 / 3), sqrt(24)]])
    )
    lower, upper = posterior.hdi(prob=1.0)
    assert lower == pytest.approx(jnp.array([[1, 10], [100, 1000]]))
    assert upper == pytest.approx(jnp.array([[5, 18], [108, 1012]]))


def test_single_component_keeps_its_dimension():
    posterior = Posterior(values=jnp.array([[1.0], [3.0], [5.0]]))

    assert posterior.mean() == pytest.approx([3.0])
    assert posterior.std() == pytest.approx([sqrt(8 / 3)])
    lower, upper = posterior.hdi(prob=1.0)
    assert lower == pytest.approx([1.0])
    assert upper == pytest.approx([5.0])


def test_single_array_draw_has_zero_spread():
    posterior = Posterior(values=jnp.array([[2.0, 4.0]]))

    assert posterior.mean() == pytest.approx([2.0, 4.0])
    assert posterior.std() == pytest.approx([0.0, 0.0])
    lower, upper = posterior.hdi()
    assert lower == pytest.approx([2.0, 4.0])
    assert upper == pytest.approx([2.0, 4.0])
