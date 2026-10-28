"""Continuous policy inference against Gaussian and quadrature reference answers."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from pylimid import (
    CategoricalPolicy,
    ChanceNode,
    ContinuousPolicy,
    DecisionNode,
    DistributionPolicy,
    InferenceError,
    InfluenceDiagram,
    Interval,
    Linear,
    LinearSoftmax,
    Real,
    UtilityNode,
    infer,
)


def _normal_rule(parents=(), parent_states=(), weights=None, bias=0.2, scale=0.4):
    family = DistributionPolicy(
        dist.Normal, Linear(), fixed_parameters={"scale": scale}
    )
    return ContinuousPolicy(
        family,
        parents,
        parent_states,
        tuple(() for _ in parents),
        Real(),
        {
            "weights": jnp.empty((0, 1)) if weights is None else jnp.asarray(weights),
            "bias": jnp.array([bias]),
        },
    )


def test_continuous_policy_conditioning_matches_a_joint_gaussian_posterior():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(DecisionNode("action", parents=("signal",), action_space=Real()))
    diagram.add_node(
        ChanceNode(
            "outcome", parents=("action",), dist=lambda action: dist.Normal(action, 0.5)
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    rule = _normal_rule(("signal",), (None,), [[0.7]])
    before = jnp.copy(rule.parameters["weights"])
    result = infer(
        diagram,
        ["signal", "action_value"],
        observed={"outcome": 0.8},
        policy={"action": rule},
        num_samples=6000,
    )
    # Prior Var(A)=0.7^2+0.4^2, Cov(S,A)=0.7, and Var(Y)=Var(A)+0.5^2.
    variance_a = 0.7**2 + 0.4**2
    variance_y = variance_a + 0.5**2
    expected_s = 0.7 / variance_y * (0.8 - 0.2)
    expected_a = 0.2 + variance_a / variance_y * (0.8 - 0.2)
    assert result["signal"].mean() == pytest.approx(expected_s, abs=0.055)
    assert result["action_value"].mean() == pytest.approx(expected_a, abs=0.04)
    assert result["signal"].std() ** 2 == pytest.approx(
        1 - 0.7**2 / variance_y, abs=0.06
    )
    assert result["action_value"].std() ** 2 == pytest.approx(
        variance_a - variance_a**2 / variance_y, abs=0.035
    )
    covariance = jnp.mean(
        (result["signal"].values - expected_s)
        * (result["action_value"].values - expected_a)
    )
    assert covariance == pytest.approx(0.7 - 0.7 * variance_a / variance_y, abs=0.035)
    assert jnp.array_equal(before, rule.parameters["weights"])


def test_unknown_category_before_continuous_action_matches_a_gaussian_mixture():
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    diagram.add_node(
        ChanceNode("category", states=labels, dist=lambda: dist.Bernoulli(0.3))
    )
    diagram.add_node(DecisionNode("action", parents=("category",), action_space=Real()))
    diagram.add_node(
        ChanceNode(
            "outcome", parents=("action",), dist=lambda action: dist.Normal(action, 0.5)
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    diagram.add_node(
        UtilityNode(
            "joint",
            parents=("category", "action"),
            values=lambda category, action: category * action,
        )
    )
    rule = _normal_rule(("category",), (labels,), [[-0.5], [1.0]], bias=0.0)
    result = infer(
        diagram,
        ["category", "action_value", "joint"],
        observed={"outcome": 0.6},
        policy={"action": rule},
        num_samples=7000,
    )
    means = np.array([-0.5, 1.0])
    variance = 0.4**2 + 0.5**2
    weights = np.array([0.7, 0.3]) * np.exp(-0.5 * (0.6 - means) ** 2 / variance)
    weights /= weights.sum()
    conditional_means = means + 0.4**2 / variance * (0.6 - means)
    assert result["category"].marginal() == pytest.approx(weights, abs=0.035)
    assert result["action_value"].mean() == pytest.approx(
        weights @ conditional_means, abs=0.045
    )
    assert result["joint"].mean() == pytest.approx(
        weights[1] * conditional_means[1], abs=0.04
    )
    assert jnp.allclose(
        result["joint"].values,
        result["category"].values * result["action_value"].values,
    )


def test_later_categorical_policy_updates_an_earlier_continuous_action():
    diagram = InfluenceDiagram()
    labels = ("off", "on")
    diagram.add_node(DecisionNode("power", action_space=Real()))
    diagram.add_node(DecisionNode("switch", parents=("power",), states=labels))
    diagram.add_node(
        ChanceNode(
            "outcome",
            parents=("switch",),
            states=labels,
            dist=lambda switch: dist.Bernoulli(jnp.array([0.1, 0.9])[switch]),
        )
    )
    diagram.add_node(
        UtilityNode("power_value", parents=("power",), values=lambda power: power)
    )
    diagram.add_node(
        UtilityNode("switch_value", parents=("switch",), values=lambda switch: switch)
    )
    first = _normal_rule(bias=0.2, scale=0.6)
    second = CategoricalPolicy(
        LinearSoftmax(),
        ("power",),
        (None,),
        ((),),
        labels,
        {"weights": jnp.array([[0.0, 1.1]]), "bias": jnp.array([0.0, -0.3])},
    )
    result = infer(
        diagram,
        ["power_value", "switch_value"],
        observed={"outcome": 1},
        policy={"power": first, "switch": second},
        num_samples=6000,
    )
    points, weights = np.polynomial.hermite.hermgauss(80)
    power = 0.2 + 0.6 * np.sqrt(2) * points
    weights /= np.sqrt(np.pi)
    p = 1 / (1 + np.exp(-(1.1 * power - 0.3)))
    likelihood = 0.1 * (1 - p) + 0.9 * p
    normalizer = weights @ likelihood
    assert result["power_value"].mean() == pytest.approx(
        weights @ (power * likelihood) / normalizer, abs=0.045
    )
    assert result["switch_value"].mean() == pytest.approx(
        weights @ (0.9 * p) / normalizer, abs=0.035
    )


@pytest.mark.parametrize("action", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("as_function", [False, True])
def test_continuous_deterministic_actions_must_stay_in_the_domain(action, as_function):
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=Interval(0.0, 1.0)))
    diagram.add_node(
        UtilityNode("payoff", parents=("power",), values=lambda power: power)
    )
    rule = (lambda: action) if as_function else action
    with pytest.raises(InferenceError, match="invalid action"):
        infer(diagram, ["payoff"], policy={"power": rule}, num_samples=2)


def test_fitted_continuous_rule_must_match_the_declared_action_domain():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=Interval(0.0, 1.0)))
    diagram.add_node(
        UtilityNode("payoff", parents=("power",), values=lambda power: power)
    )
    with pytest.raises(InferenceError, match="incompatible"):
        infer(diagram, ["payoff"], policy={"power": _normal_rule()}, num_samples=2)


def test_continuous_actions_do_not_accept_categorical_action_tables():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=Real()))
    diagram.add_node(
        UtilityNode("payoff", parents=("power",), values=lambda power: power)
    )
    with pytest.raises(InferenceError, match="ActionTable.*categorical"):
        infer(diagram, ["payoff"], policy={"power": {(): 0}}, num_samples=2)


def test_continuous_function_policies_retain_the_evidence_restriction():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=Real()))
    diagram.add_node(
        ChanceNode(
            "outcome", parents=("power",), dist=lambda power: dist.Normal(power, 1.0)
        )
    )
    with pytest.raises(NotImplementedError, match="forward evaluation only"):
        infer(
            diagram,
            ["outcome"],
            observed={"outcome": 0.0},
            policy={"power": lambda: 0.5},
            num_samples=2,
        )


def test_an_observed_continuous_information_parent_retains_fitted_shape_checks():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(DecisionNode("power", parents=("signal",), action_space=Real()))
    diagram.add_node(
        UtilityNode("payoff", parents=("power",), values=lambda power: power)
    )
    rule = _normal_rule(("signal",), (None,), [[0.7]])
    incompatible = replace(rule, parent_shapes=((2,),))
    with pytest.raises(ValueError, match="trailing shape"):
        infer(
            diagram,
            ["payoff"],
            observed={"signal": 0.0},
            policy={"power": incompatible},
            num_samples=2,
        )


def test_truncated_normal_policy_conditioning_matches_bounded_quadrature():
    family = DistributionPolicy(
        dist.TruncatedNormal,
        Linear(),
        fixed_parameters={"scale": 0.4, "low": 0.0, "high": 1.0},
    )
    rule = ContinuousPolicy(
        family,
        (),
        (),
        (),
        Interval(0.0, 1.0),
        {"weights": jnp.empty((0, 1)), "bias": jnp.array([0.2])},
    )
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("action", action_space=Interval(0.0, 1.0)))
    diagram.add_node(
        ChanceNode(
            "outcome", parents=("action",), dist=lambda action: dist.Normal(action, 0.2)
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    diagram.add_node(
        UtilityNode("squared", parents=("action",), values=lambda action: action**2)
    )
    result = infer(
        diagram,
        ["action_value", "squared"],
        observed={"outcome": 0.6},
        policy={"action": rule},
        num_samples=5000,
    )
    points, weights = np.polynomial.legendre.leggauss(120)
    points = (points + 1) / 2
    weights = weights * np.exp(
        -0.5 * ((points - 0.2) / 0.4) ** 2 - 0.5 * ((0.6 - points) / 0.2) ** 2
    )
    weights /= weights.sum()
    assert result["action_value"].mean() == pytest.approx(weights @ points, abs=0.025)
    assert result["squared"].mean() == pytest.approx(weights @ points**2, abs=0.025)
    assert jnp.all(
        (result["action_value"].values >= 0) & (result["action_value"].values <= 1)
    )
    assert jnp.allclose(result["squared"].values, result["action_value"].values ** 2)


def test_categorical_information_and_beta_actions_match_conjugate_mixture():
    labels = ("off", "on")
    family = DistributionPolicy(dist.Beta, Linear())
    rule = ContinuousPolicy(
        family,
        ("category",),
        (labels,),
        ((),),
        Interval(0.0, 1.0),
        {"weights": jnp.log(jnp.array([[2.0, 4.0], [4.0, 2.0]])), "bias": jnp.zeros(2)},
    )
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode("category", states=labels, dist=lambda: dist.Bernoulli(0.4))
    )
    diagram.add_node(
        DecisionNode("action", parents=("category",), action_space=Interval(0.0, 1.0))
    )
    diagram.add_node(
        ChanceNode(
            "success",
            parents=("action",),
            states=labels,
            dist=lambda action: dist.Bernoulli(action),
        )
    )
    diagram.add_node(
        UtilityNode("action_value", parents=("action",), values=lambda action: action)
    )
    diagram.add_node(
        UtilityNode(
            "joint",
            parents=("category", "action"),
            values=lambda category, action: category * action,
        )
    )
    result = infer(
        diagram,
        ["category", "action_value", "joint"],
        observed={"success": 1},
        policy={"action": rule},
        num_samples=6000,
    )
    # Component evidence is alpha/(alpha+beta); each posterior adds one to alpha.
    weights = np.array([0.6 * 2 / 6, 0.4 * 4 / 6])
    weights /= weights.sum()
    means = np.array([3 / 7, 5 / 7])
    assert result["category"].marginal() == pytest.approx(weights, abs=0.035)
    assert result["action_value"].mean() == pytest.approx(weights @ means, abs=0.03)
    assert result["joint"].mean() == pytest.approx(weights[1] * means[1], abs=0.03)
