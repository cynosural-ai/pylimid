"""Validate trained oil policies against analytic utilities."""

import jax
import jax.numpy as jnp
import pytest

from examples.oil_policy import build_diagram, exact_policy_value
from pylimid import TabularPolicy, TabularSoftmax, solve


def test_oil_exact_evaluation_matches_analytic_deterministic_and_uniform_policies():
    buy = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 0,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 0,
    }
    assert exact_policy_value({"test": {(): 1}, "buy": buy}) == pytest.approx(721.0)
    assert exact_policy_value({"test": {(): 0}, "buy": buy}) == pytest.approx(696.95)
    uniform = {
        "test": TabularPolicy(
            TabularSoftmax(), (), (), (), ("do", "not_do"), jnp.zeros(2)
        ),
        "buy": TabularPolicy(
            TabularSoftmax(),
            ("test", "result"),
            (("do", "not_do"), ("pass", "fail", "no_result")),
            ((), ()),
            ("buy", "not_buy"),
            jnp.zeros((2, 3, 2)),
        ),
    }
    # Half the mean purchase reward plus half the outside option, less half a test.
    assert exact_policy_value(uniform) == pytest.approx(0.5 * 721 + 0.5 * 350 - 15)


def test_trained_oil_policy_approaches_analytic_optimum():
    fitted = solve(
        build_diagram(),
        num_steps=200,
        num_samples=2000,
        rng_key=jax.random.PRNGKey(0),
    )
    assert 0 < 721.0 - exact_policy_value(fitted.policy) < 2.0
