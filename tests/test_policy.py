"""Evaluate deterministic function policies through sampling and public inference."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    UtilityNode,
    infer,
)
from pylimid._numpyro.samplers import samples


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


def test_function_policy_uses_continuous_observation():
    diagram = _signal_diagram()
    draws = samples(
        diagram.snapshot(), policy={"action": _positive_signal}, num_samples=4000
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
def test_function_must_accept_declared_information_parents(rule):
    with pytest.raises(InferenceError, match="information parents"):
        samples(_signal_diagram().snapshot(), policy={"action": rule})


@pytest.mark.parametrize("value", [0.5, True, jnp.array([0, 1])])
def test_function_must_return_scalar_integer_action(value):
    with pytest.raises(InferenceError, match="scalar integer action"):
        samples(_signal_diagram().snapshot(), policy={"action": lambda signal: value})


@pytest.mark.parametrize("value", [-1, 2])
def test_sampled_function_actions_must_belong_to_action_space(value):
    with pytest.raises(InferenceError, match="invalid action"):
        samples(_signal_diagram().snapshot(), policy={"action": lambda signal: value})


def test_function_policies_reject_chance_evidence_in_backend():
    with pytest.raises(NotImplementedError, match="forward evaluation only"):
        samples(
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
