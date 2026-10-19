"""Posterior summaries for scalar and array-valued draws."""

from math import sqrt

import jax
import jax.numpy as jnp
import pytest

from pylimid import Posterior


def test_integer_numerical_draws_are_not_categories():
    posterior = Posterior(values=jnp.array([0, 2, 4, 6]))

    assert not posterior.is_categorical
    assert posterior.mean() == 3.0
    with pytest.raises(ValueError, match="requires a categorical"):
        posterior.marginal()


def test_scalar_summaries_return_zero_dimensional_arrays():
    posterior = Posterior(values=jnp.array([0.0, 2.0, 4.0, 6.0]))

    assert posterior.mean() == 3.0
    assert posterior.std() == pytest.approx(sqrt(5))
    assert posterior.hdi(prob=0.75) == (0.0, 4.0)
    assert posterior.hdi(prob=1.0) == (0.0, 6.0)
    assert all(
        isinstance(value, jax.Array) and value.shape == ()
        for value in (posterior.mean(), posterior.std(), *posterior.hdi())
    )


def test_discrete_marginal_includes_states_without_draws():
    posterior = Posterior(values=jnp.array([0, 1, 1, 3]), states=("a", "b", "c", "d"))

    probabilities = posterior.marginal()

    assert isinstance(probabilities, jax.Array)
    assert probabilities.shape == (4,)
    assert probabilities == pytest.approx([0.25, 0.5, 0.0, 0.25])


@pytest.mark.parametrize("invalid_index", [-1, 2])
def test_discrete_marginal_rejects_indices_outside_states(invalid_index):
    posterior = Posterior(values=jnp.array([0, invalid_index]), states=("a", "b"))

    with pytest.raises(ValueError, match="state indices from 0 to 1"):
        posterior.marginal()


def test_discrete_marginal_does_not_truncate_fractional_indices():
    posterior = Posterior(values=jnp.array([0.5, 1.0]), states=("a", "b"))

    with pytest.raises(TypeError):
        posterior.marginal()


def test_summaries_preserve_float64_precision():
    with jax.enable_x64():
        posterior = Posterior(
            values=jnp.array([1e9, 1e9 + 2, 1e9 + 4], dtype=jnp.float64)
        )

        assert posterior.mean().dtype == jnp.float64
        assert posterior.mean() == 1e9 + 2
        assert posterior.std() == pytest.approx(sqrt(8 / 3), rel=1e-12)
        lower, upper = posterior.hdi(prob=1.0)
        assert lower.dtype == upper.dtype == jnp.float64
        assert lower == 1e9
        assert upper == 1e9 + 4


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
