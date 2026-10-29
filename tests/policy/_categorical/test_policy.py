"""Fitted categorical rules, information encoding, and action probabilities."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from pylimid.policy import CategoricalPolicy, LinearSoftmax, TabularSoftmax


def test_categorical_features_are_one_hot_with_numerical_features_kept_separate():
    rule = CategoricalPolicy(
        LinearSoftmax(),
        ("category", "signal"),
        (("red", "green", "blue"), None),
        ((), ()),
        ("off", "on"),
        {
            "weights": jnp.array([[0.0, 2.0], [0.0, -2.0], [0.0, 2.0], [0.0, 1.0]]),
            "bias": jnp.zeros(2),
        },
    )
    for index, expected in enumerate(
        [jax.nn.sigmoid(2.0), jax.nn.sigmoid(-2.0), jax.nn.sigmoid(2.0)]
    ):
        assert rule.probabilities(signal=jnp.array(0.0), category=jnp.array(index))[
            1
        ] == pytest.approx(expected)
    assert rule.probabilities(signal=jnp.array(1.0), category=jnp.array(0))[
        1
    ] == pytest.approx(jax.nn.sigmoid(3.0))
    with pytest.raises(TypeError, match="Unexpected information"):
        rule.probabilities(
            signal=jnp.array(0.0), category=jnp.array(0), hidden=jnp.array(1.0)
        )


def test_tabular_rows_preserve_parent_order_and_have_independent_parameters():
    logits = jnp.arange(18, dtype=float).reshape(3, 2, 3) / 10
    rule = CategoricalPolicy(
        family=TabularSoftmax(),
        parents=("right", "left"),
        parent_states=(("red", "green", "blue"), ("no", "yes")),
        parent_shapes=((), ()),
        states=("low", "medium", "high"),
        parameters={"logits": logits},
    )
    information = {"left": jnp.array(1), "right": jnp.array(2)}
    probabilities = rule.probabilities(**information)
    assert probabilities == pytest.approx(jax.nn.softmax(logits[2, 1]))
    assert jnp.exp(rule.log_prob(jnp.array(2), **information)) == pytest.approx(
        probabilities[2]
    )
    key = jax.random.PRNGKey(4)
    assert rule.sample(key, **information) == rule.sample(key, **information)
    gradient = jax.grad(
        lambda scores: replace(
            rule, parameters={**rule.parameters, "logits": scores}
        ).log_prob(jnp.array(2), **information)
    )(logits)
    assert gradient[2, 1] == pytest.approx(jnp.array([0.0, 0.0, 1.0]) - probabilities)
    assert jnp.count_nonzero(gradient.at[2, 1].set(0)) == 0
    with pytest.raises(TypeError, match="Unexpected information"):
        rule.probabilities(**information, hidden=jnp.array(0))
    with pytest.raises(ValueError, match="must have shape"):
        rule.probabilities(left=jnp.array([1]), right=jnp.array(2))
