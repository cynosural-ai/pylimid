"""Policy execution, joint action draws, and fitted metadata validation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid._numpyro.samplers import samples
from pylimid.errors import InferenceError
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode, DecisionNode
from pylimid.inference import infer
from pylimid.policy import CategoricalPolicy, TabularSoftmax
from pylimid.solver.api import _initialize_policy
from tests._diagrams import signal_diagram, uniform_rule, xor_diagram
from tests._numpyro.diagrams import rain_node


@pytest.mark.parametrize(
    "rule, message",
    [
        ({(0,): 0}, "every information-set assignment"),
        ({(0,): 0, (1,): 1, (2,): 0}, "every information-set assignment"),
        ({(0,): -1, (1,): 1}, "invalid action"),
        ({(0,): 0, (1,): 2}, "invalid action"),
        ({(0,): 0, (1,): 0.5}, "invalid action"),
    ],
)
def test_backend_rejects_invalid_contingent_policy(rule, message):
    diagram = InfluenceDiagram()
    diagram.add_node(rain_node())
    diagram.add_node(
        DecisionNode(name="action", parents=("rain",), states=("no", "yes"))
    )
    with pytest.raises(InferenceError, match=message):
        samples(diagram.snapshot(), policy={"action": rule})


def test_evaluation_preserves_stochastic_action_descendant_correlation():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("action", states=("off", "on")))
    diagram.add_node(DecisionNode("next", parents=("action",), states=("off", "on")))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("action", "next"),
            dist=lambda action, next: dist.Delta(action + 2 * next),
        )
    )
    policy = {"action": uniform_rule(), "next": {(0,): 1, (1,): 0}}
    draws = samples(diagram.snapshot(), policy=policy, num_samples=1000)
    assert jnp.mean(draws["action"]) == pytest.approx(0.5, abs=0.05)
    assert jnp.array_equal(draws["next"], 1 - draws["action"])
    assert jnp.array_equal(draws["outcome"], 2 - draws["action"])
    posterior = samples(
        diagram.snapshot(), policy=policy, observed={"outcome": 1}, num_samples=100
    )
    assert jnp.array_equal(posterior["action"], jnp.ones(100, dtype=int))
    assert jnp.array_equal(posterior["next"], jnp.zeros(100, dtype=int))
    assert jnp.array_equal(posterior["outcome"], jnp.ones(100, dtype=int))


@pytest.mark.parametrize(
    "changes",
    [
        {"states": ("on", "off")},
        {"parents": ("other",)},
        {"parent_states": (("low", "high"),)},
    ],
)
def test_fitted_policy_rejects_changed_action_or_information_states(changes):
    diagram = signal_diagram()
    rule = replace(uniform_rule(("signal",), (None,), 1), **changes)
    with pytest.raises(InferenceError, match="incompatible"):
        samples(diagram.snapshot(), policy={"action": rule})


def test_tabular_root_policy_preserves_stochastic_action_and_descendant_correlation():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("first", states=("off", "on")))
    diagram.add_node(DecisionNode("second", parents=("first",), states=("off", "on")))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("first", "second"),
            dist=lambda first, second: dist.Delta(first + 2 * second),
        )
    )
    rule = CategoricalPolicy(
        TabularSoftmax(),
        (),
        (),
        (),
        ("off", "on"),
        {"logits": jnp.log(jnp.array([0.3, 0.7]))},
    )
    policy = {"first": rule, "second": {(0,): 1, (1,): 0}}
    draws = samples(diagram.snapshot(), policy=policy, num_samples=5000)
    assert jnp.mean(draws["first"]) == pytest.approx(0.7, abs=0.02)
    assert jnp.array_equal(draws["second"], 1 - draws["first"])
    assert jnp.array_equal(draws["outcome"], 2 - draws["first"])
    posterior = samples(
        diagram.snapshot(), policy=policy, observed={"outcome": 1}, num_samples=100
    )
    assert jnp.array_equal(posterior["first"], jnp.ones(100, dtype=int))
    assert jnp.array_equal(posterior["second"], jnp.zeros(100, dtype=int))
    assert jnp.array_equal(posterior["outcome"], jnp.ones(100, dtype=int))
    result = infer(
        diagram, ["outcome"], policy=policy, observed={"outcome": 1}, num_samples=100
    )
    assert jnp.array_equal(result["outcome"].values, posterior["outcome"])


@pytest.mark.parametrize(
    "changes",
    [
        {"states": ("on", "off")},
        {"parents": ("other", "right")},
        {"parent_states": (("yes", "no"), ("no", "yes"))},
    ],
)
def test_tabular_policy_rejects_changed_information_and_action_labels(changes):
    diagram = xor_diagram(TabularSoftmax())
    rules = _initialize_policy(diagram.snapshot(), jax.random.PRNGKey(0))
    rules["action"] = replace(rules["action"], **changes)
    with pytest.raises(InferenceError, match="incompatible"):
        samples(diagram.snapshot(), policy=rules)
