"""Evaluate deterministic function policies through the public sampling paths."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from pylimid import (
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    UtilityNode,
    infer,
)
from pylimid.numpyro import samples, to_model
from pylimid.solver import Solution


def _signal_diagram():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(DecisionNode("action", parents=("signal",), states=("off", "on")))
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: signal * action,
        )
    )
    return diagram


def _positive_signal(signal):
    return jnp.where(signal > 0, 1, 0)


@pytest.mark.parametrize("backend", [samples, to_model])
def test_function_policy_uses_continuous_observation(backend):
    diagram = _signal_diagram()
    result = backend(diagram.snapshot(), policy={"action": _positive_signal})
    draws = (
        Predictive(result, num_samples=4000)(jax.random.PRNGKey(0))
        if backend is to_model
        else result
    )
    assert draws["action"].shape == draws["signal"].shape
    assert jnp.array_equal(draws["action"], draws["signal"] > 0)


def test_infer_evaluates_function_policy_utility():
    result = infer(
        _signal_diagram(),
        ["signal", "payoff"],
        policy={"action": _positive_signal},
        num_samples=10000,
    )
    signal = result["signal"].values
    assert jnp.array_equal(result["payoff"].values, jnp.maximum(signal, 0))
    assert result["payoff"].mean() == pytest.approx(0.25, abs=0.015)


def test_functions_mix_with_tables_and_fixed_actions_by_parent_name():
    diagram = _signal_diagram()
    diagram.add_node(
        ChanceNode("category", states=("low", "high"), dist=lambda: dist.Bernoulli(0.5))
    )
    diagram.add_node(DecisionNode("previous", states=("off", "on")))
    diagram["action"].parents = ("category", "previous", "signal")
    diagram.add_node(DecisionNode("next", parents=("action",), states=("off", "on")))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("action", "next"),
            dist=lambda action, next: dist.Delta(action + 2 * next),
        )
    )

    def rule(*, signal, previous, category):
        return jnp.where(signal > 0, 1, 0) ^ category ^ previous

    result = infer(
        diagram,
        ["signal", "category", "outcome"],
        policy={"next": {(0,): 1, (1,): 0}, "action": rule, "previous": 1},
        num_samples=200,
    )
    expected = (result["signal"].values > 0).astype(int) ^ result["category"].values ^ 1
    assert jnp.array_equal(result["outcome"].values, expected + 2 * (1 - expected))


def test_function_with_no_information_parents_returns_constant_action():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("action", states=("off", "on")))
    draws = samples(diagram.snapshot(), policy={"action": lambda: 1}, num_samples=30)
    assert jnp.array_equal(draws["action"], jnp.ones(30, dtype=int))


@pytest.mark.parametrize(
    "rule", [lambda hidden: 0, lambda **kwargs: 0, lambda signal, hidden: 0]
)
@pytest.mark.parametrize("backend", [samples, to_model])
def test_function_must_accept_declared_information_parents(backend, rule):
    with pytest.raises(InferenceError, match="information parents"):
        backend(_signal_diagram().snapshot(), policy={"action": rule})


@pytest.mark.parametrize("value", [0.5, True, jnp.array([0, 1])])
@pytest.mark.parametrize("backend", [samples, to_model])
def test_function_must_return_scalar_integer_action(backend, value):
    with pytest.raises(InferenceError, match="scalar integer action"):
        result = backend(
            _signal_diagram().snapshot(), policy={"action": lambda signal: value}
        )
        if backend is to_model:
            Predictive(result, num_samples=10)(jax.random.PRNGKey(0))


@pytest.mark.parametrize("value", [-1, 2])
def test_sampled_function_actions_must_belong_to_action_space(value):
    with pytest.raises(InferenceError, match="invalid action"):
        samples(_signal_diagram().snapshot(), policy={"action": lambda signal: value})


@pytest.mark.parametrize("backend", [samples, to_model])
def test_function_policies_reject_chance_evidence_in_backend(backend):
    with pytest.raises(NotImplementedError, match="forward evaluation only"):
        backend(
            _signal_diagram().snapshot(),
            policy={"action": _positive_signal},
            observed={"signal": 0.5},
        )


def test_infer_rejects_chance_evidence_with_relevant_function_policy():
    with pytest.raises(NotImplementedError, match="forward evaluation only"):
        infer(
            _signal_diagram(),
            ["payoff"],
            policy={"action": _positive_signal},
            observed={"signal": 0.5},
        )


def test_infer_ignores_irrelevant_function_policy_with_evidence():
    result = infer(
        _signal_diagram(),
        ["signal"],
        policy={"action": _positive_signal},
        observed={"signal": 0.5},
        num_samples=20,
    )
    assert jnp.all(result["signal"].values == 0.5)


def test_function_policy_can_be_rendered_without_enumerating_observations():
    solution = Solution(
        policy={"action": _positive_signal}, expected_utility=0.25, method="scan"
    )
    rendered = solution.render(_signal_diagram())
    assert "action (signal)" in rendered
    assert "deterministic function policy" in rendered
