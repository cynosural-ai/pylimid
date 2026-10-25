"""Validate the oil comparison against analytic strategies and exact inference."""

import jax
import jax.numpy as jnp
import pytest

from examples.oil_policy_comparison import (
    build_pylimid,
    deterministic_policy,
    exact_optimum,
    exact_policy_value,
    solve_pyagrum,
)
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


def test_oil_three_solvers_agree_under_exact_policy_evaluation():
    optimum = exact_optimum()
    pyagrum_value, pyagrum_policy = solve_pyagrum()
    assert optimum == pytest.approx(721.0)
    assert pyagrum_value == pytest.approx(optimum)
    assert exact_policy_value(pyagrum_policy) == pytest.approx(optimum)
    with pytest.warns(UserWarning, match="Decision 'buy'"):
        backward = solve(build_pylimid(), method="backward_induction", num_samples=4000)
    assert exact_policy_value(backward.policy) == pytest.approx(optimum)
    fitted = solve(
        build_pylimid(trainable=True),
        method="policy_gradient",
        num_steps=200,
        num_samples=2000,
        rng_key=jax.random.PRNGKey(0),
    )
    assert 0 < optimum - exact_policy_value(fitted.policy) < 2.0
    assert exact_policy_value(deterministic_policy(fitted.policy)) == pytest.approx(
        optimum
    )
