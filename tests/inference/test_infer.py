"""Tests for :func:`pylimid.inference.infer`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import InferenceError, Interval, Posterior, Real, infer, solve
from pylimid._numpyro.samplers import samples
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes.chance_node import ChanceNode
from pylimid.graph.nodes.decision_node import DecisionNode
from pylimid.graph.nodes.utility_node import UtilityNode

# --- helpers -----------------------------------------------------------------


def _rain_wet_grass() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="wet_grass",
            parents=("rain",),
            states=("dry", "wet"),
            dist=lambda rain: dist.Categorical(
                probs=jnp.array([[0.80, 0.20], [0.05, 0.95]])[rain]
            ),
        )
    )
    return diag


def _continuous_diagram() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(loc=5.0, scale=2.0)))
    return diag


def _mixed_diagram() -> InfluenceDiagram:
    """Rain (discrete root) and temperature (continuous root)."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="temp",
            dist=lambda: dist.Normal(loc=20.0, scale=3.0),
        )
    )
    return diag


def _decision_diagram() -> InfluenceDiagram:
    """A valid influence diagram with a decision and a utility node."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        DecisionNode(name="umbrella", parents=("rain",), states=("no", "yes"))
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: float(rain == umbrella),
        )
    )
    return diag


def _umbrella_diagram() -> InfluenceDiagram:
    """``rain -> umbrella (decision) -> wet``; utility on ``(rain, umbrella)``."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        DecisionNode(name="umbrella", parents=("rain",), states=("no", "yes"))
    )
    diag.add_node(
        ChanceNode(
            name="wet",
            parents=("rain", "umbrella"),
            states=("dry", "wet"),
            dist=lambda rain, umbrella: dist.Categorical(
                # Single indexing op: numpyro enumeration mis-weights
                # chained ``[rain][umbrella]`` indexing.
                probs=jnp.array(
                    [
                        [[0.90, 0.10], [0.70, 0.30]],  # rain=0
                        [[0.40, 0.60], [0.05, 0.95]],  # rain=1
                    ]
                )[rain, umbrella]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain", "umbrella"),
            values=lambda rain, umbrella: float(rain == umbrella),
        )
    )
    return diag


# --- results -----------------------------------------------------------------


def test_discrete_prior():
    result = infer(_rain_wet_grass(), ["rain"])
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].is_categorical
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_discrete_posterior_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].marginal() == pytest.approx([0.457, 0.543], abs=0.04)


def test_result_keys_match_query():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_probabilities_sum_to_one():
    result = infer(_rain_wet_grass(), ["rain", "wet_grass"])
    assert isinstance(result["rain"], Posterior)
    assert isinstance(result["wet_grass"], Posterior)
    assert result["rain"].marginal().sum() == pytest.approx(1.0)
    assert result["wet_grass"].marginal().sum() == pytest.approx(1.0)


def test_probabilities_sum_to_one_with_observed():
    result = infer(_rain_wet_grass(), ["rain"], observed={"wet_grass": 1})
    assert isinstance(result["rain"], Posterior)
    probabilities = result["rain"].marginal()
    assert isinstance(probabilities, jax.Array)
    assert probabilities.shape == (2,)
    assert probabilities.sum() == pytest.approx(1.0)


def test_continuous_posterior_keeps_raw_draws():
    """Continuous posteriors store raw draws and carry no states."""
    result = infer(_continuous_diagram(), ["x"], num_samples=2000)
    assert isinstance(result["x"], Posterior)
    assert result["x"].states is None
    assert not result["x"].is_categorical
    assert isinstance(result["x"].values, jax.Array)
    assert result["x"].values.shape == (2000,)
    assert jnp.issubdtype(result["x"].values.dtype, jnp.floating)
    assert result["x"].mean() == pytest.approx(5.0, abs=0.2)
    assert result["x"].std() == pytest.approx(2.0, abs=0.15)
    lo, hi = result["x"].hdi()
    assert lo < 5.0 < hi


def test_continuous_draws_preserve_float64_dtype():
    with jax.enable_x64():
        diagram = InfluenceDiagram()
        diagram.add_node(
            ChanceNode(
                name="x",
                dist=lambda: dist.Normal(jnp.asarray(1e9, dtype=jnp.float64), 1.0),
            )
        )

        posterior = infer(diagram, ["x"], num_samples=20)["x"]

        assert posterior.values.dtype == jnp.float64
        assert posterior.mean().dtype == jnp.float64


def test_dirichlet_prior_can_be_queried_with_a_scalar_child():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="weights", dist=lambda: dist.Dirichlet(jnp.array([2.0, 3.0, 5.0]))
        )
    )
    diagram.add_node(
        ChanceNode(
            name="outcome",
            parents=("weights",),
            states=("a", "b", "c"),
            dist=lambda weights: dist.Categorical(probs=weights),
        )
    )

    result = infer(diagram, ["weights", "outcome"], num_samples=1000)
    weights = result["weights"]

    assert weights.states is None
    assert not weights.is_categorical
    assert isinstance(weights.values, jax.Array)
    assert weights.values.shape == (1000, 3)
    assert jnp.allclose(weights.values.sum(axis=1), 1.0)
    assert weights.mean() == pytest.approx([0.2, 0.3, 0.5], abs=0.03)
    assert isinstance(result["outcome"].values, jax.Array)
    assert result["outcome"].values.shape == (1000,)
    assert jnp.issubdtype(result["outcome"].values.dtype, jnp.integer)
    assert result["outcome"].marginal() == pytest.approx([0.2, 0.3, 0.5], abs=0.05)


def test_dirichlet_posterior_updates_and_keeps_joint_draw_order():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="weights", dist=lambda: dist.Dirichlet(jnp.array([2.0, 3.0, 5.0]))
        )
    )
    diagram.add_node(
        ChanceNode(
            name="outcome",
            parents=("weights",),
            states=("a", "b", "c"),
            dist=lambda weights: dist.Categorical(probs=weights),
        )
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("weights", "outcome"),
            values=lambda weights, outcome: weights[outcome],
        )
    )

    result = infer(
        diagram,
        ["weights", "outcome", "payoff"],
        observed={"outcome": 1},
        num_samples=1000,
    )
    weights = result["weights"]
    assert isinstance(weights.values, jax.Array)
    assert weights.values.shape == (1000, 3)
    # Dirichlet-Categorical conjugacy gives concentration [2, 4, 5].
    concentration = jnp.array([2.0, 4.0, 5.0])
    assert weights.mean() == pytest.approx(concentration / 11, abs=0.03)
    assert weights.std() == pytest.approx(
        jnp.sqrt(concentration * (11 - concentration) / (11**2 * 12)), abs=0.02
    )
    lower, upper = weights.hdi()
    assert isinstance(lower, jax.Array)
    assert isinstance(upper, jax.Array)
    assert lower.shape == upper.shape == (3,)
    assert jnp.all(lower < concentration / 11)
    assert jnp.all(upper > concentration / 11)
    assert jnp.issubdtype(result["outcome"].values.dtype, jnp.integer)
    assert jnp.all(result["outcome"].values == 1)
    assert jnp.array_equal(result["payoff"].values, weights.values[:, 1])


@pytest.mark.parametrize("node_shape", [(1,), (2, 2)])
def test_array_queries_preserve_event_dimensions(node_shape):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="x",
            dist=lambda: dist.Normal(jnp.ones(node_shape), 0.1).to_event(
                len(node_shape)
            ),
        )
    )

    posterior = infer(diagram, ["x"], num_samples=50)["x"]

    assert isinstance(posterior.values, jax.Array)
    assert posterior.values.shape == (50, *node_shape)
    assert posterior.mean() == pytest.approx(jnp.ones(node_shape), abs=0.1)
    assert posterior.std() == pytest.approx(jnp.full(node_shape, 0.1), abs=0.05)
    lower, upper = posterior.hdi()
    assert isinstance(lower, jax.Array)
    assert isinstance(upper, jax.Array)
    assert lower.shape == upper.shape == node_shape


def test_mixed_result_shape_follows_variable_type():
    """One Posterior type; states distinguish categories from numerical values."""
    result = infer(_mixed_diagram(), ["rain", "temp"], num_samples=2000)
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].states == ("no", "yes")
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)
    assert isinstance(result["temp"], Posterior)
    assert result["temp"].states is None
    assert len(result["temp"].values) == 2000  # raw MCMC draws
    assert result["temp"].mean() == pytest.approx(20.0, abs=0.3)


def test_discrete_posterior_stores_state_index_draws():
    """Discrete draws are integer state indices; the vector is derived."""
    result = infer(_rain_wet_grass(), ["rain"])
    assert isinstance(result["rain"].values, jax.Array)
    assert jnp.issubdtype(result["rain"].values.dtype, jnp.integer)
    assert jnp.all((result["rain"].values == 0) | (result["rain"].values == 1))
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_summaries_on_categorical_raise():
    """Index draws are not moments: mean/std/hdi refuse categorical posteriors."""
    result = infer(_rain_wet_grass(), ["rain"])
    for method in ("mean", "std", "hdi"):
        with pytest.raises(ValueError, match="requires a numerical"):
            getattr(result["rain"], method)()


def test_marginal_on_continuous_raises():
    result = infer(_continuous_diagram(), ["x"])
    with pytest.raises(ValueError, match="marginal\\(\\) requires a categorical"):
        result["x"].marginal()


def test_hdi_invalid_coverage_raises():
    result = infer(_continuous_diagram(), ["x"])
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="coverage must be in"):
            result["x"].hdi(prob=bad)


# --- errors ------------------------------------------------------------------


def test_unbound_decision_upstream_of_query_raises():
    with pytest.raises(InferenceError, match="unbound decision.*umbrella"):
        infer(_umbrella_diagram(), ["wet"])


def test_unbound_decision_upstream_of_evidence_raises():
    with pytest.raises(InferenceError, match="unbound decision.*umbrella"):
        infer(_umbrella_diagram(), ["rain"], observed={"wet": 1})


def _contingent_policy_diagram() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="state",
            states=("low", "high"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(DecisionNode(name="first", parents=("state",), states=("a", "b")))
    diag.add_node(DecisionNode(name="second", parents=("first",), states=("a", "b")))
    diag.add_node(
        ChanceNode(
            name="evidence",
            parents=("first",),
            states=("no", "yes"),
            dist=lambda first: dist.Categorical(
                probs=jnp.array([[0.9, 0.1], [0.1, 0.9]])[first]
            ),
        )
    )
    diag.add_node(
        UtilityNode(
            name="payoff",
            parents=("first", "second"),
            values=lambda first, second: first + second,
        )
    )
    return diag


@pytest.mark.parametrize("reverse_policy", [False, True])
def test_contingent_policy_is_evaluated_in_forward_sampling(reverse_policy):
    diagram = _contingent_policy_diagram()
    policy = {"first": {(0,): 0, (1,): 1}, "second": {(0,): 1, (1,): 0}}
    if reverse_policy:
        policy = dict(reversed(list(policy.items())))
    payoff = infer(diagram, ["payoff"], policy=policy, num_samples=5000)["payoff"]
    assert jnp.all(payoff.values == 1)


@pytest.mark.parametrize("reverse_policy", [False, True])
@pytest.mark.parametrize("invert_first", [False, True])
def test_contingent_policy_keeps_latent_information_parent_conditionable(
    reverse_policy, invert_first
):
    diagram = _contingent_policy_diagram()
    policy = {"first": {(0,): 0, (1,): 1}, "second": {(0,): 1, (1,): 0}}
    if invert_first:
        policy["first"] = {(0,): 1, (1,): 0}
    if reverse_policy:
        policy = dict(reversed(list(policy.items())))
    result = infer(
        diagram, ["state", "payoff"], policy=policy, observed={"evidence": 1}
    )
    expected = [0.9, 0.1] if invert_first else [0.1, 0.9]
    assert result["state"].marginal() == pytest.approx(expected, abs=0.04)
    assert jnp.all(result["payoff"].values == 1)


def test_contingent_policy_accepts_solve_result_for_forward_evaluation():
    from pylimid import TabularSoftmax

    diagram = _contingent_policy_diagram()
    diagram["first"].policy_family = TabularSoftmax()
    diagram["second"].policy_family = TabularSoftmax()
    solution = solve(diagram, num_steps=200, num_samples=500)
    payoff = infer(diagram, ["payoff"], policy=solution.policy, num_samples=2000)
    assert payoff["payoff"].mean() > 1.95


@pytest.mark.parametrize("reverse_policy", [False, True])
def test_contingent_policy_with_mixed_nuts_posterior(reverse_policy):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="state",
            states=("low", "high"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diagram.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(DecisionNode(name="action", parents=("state",), states=("a", "b")))
    diagram.add_node(
        DecisionNode(name="second", parents=("action",), states=("a", "b"))
    )
    diagram.add_node(
        ChanceNode(
            name="measurement",
            parents=("state", "action", "x"),
            dist=lambda state, action, x: dist.Normal(x + state + action, 1.0),
        )
    )
    diagram.add_node(
        UtilityNode(name="payoff", parents=("second",), values=lambda second: second)
    )
    policy = {"action": {(0,): 0, (1,): 1}, "second": {(0,): 1, (1,): 0}}
    if reverse_policy:
        policy = dict(reversed(list(policy.items())))
    result = infer(
        diagram,
        ["state", "x", "payoff"],
        policy=policy,
        observed={"measurement": 2.0},
        num_samples=500,
    )
    # The two likelihoods are Normal(0, sqrt(2)) and Normal(2, sqrt(2)).
    # At measurement=2, P(state=high) = 1 / (1 + exp(-1)).
    assert len(result["x"].values) == 500
    assert result["state"].marginal()[1] == pytest.approx(0.731, abs=0.06)
    assert result["x"].mean() == pytest.approx(0.269, abs=0.15)
    assert jnp.array_equal(result["payoff"].values, 1 - result["state"].values)


def test_contingent_policy_can_follow_a_fixed_decision():
    result = infer(
        _contingent_policy_diagram(),
        ["payoff"],
        policy={"first": 1, "second": {(0,): 1, (1,): 0}},
        observed={"evidence": 1},
    )
    assert jnp.all(result["payoff"].values == 1)


def test_contingent_policy_table_must_be_complete_and_valid():
    diagram = _contingent_policy_diagram()
    with pytest.raises(InferenceError, match="every information-set assignment"):
        infer(diagram, ["payoff"], policy={"first": {(0,): 0}, "second": {(0,): 0}})
    with pytest.raises(InferenceError, match="invalid action"):
        infer(
            diagram,
            ["payoff"],
            policy={"first": {(0,): 3, (1,): 0}, "second": {(0,): 0, (1,): 0}},
        )


def test_unknown_observed_name_raises():
    with pytest.raises(KeyError, match="bogus"):
        infer(_rain_wet_grass(), ["rain"], observed={"bogus": 1})


# --- sampling options ---------------------------------------------------------


def test_num_samples_sets_the_number_of_draws():
    for diagram, query in ((_rain_wet_grass(), "rain"), (_continuous_diagram(), "x")):
        result = infer(diagram, [query], num_samples=500)
        assert len(result[query].values) == 500


def test_num_samples_with_nuts():
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0.0, 1.0)))
    diag.add_node(
        ChanceNode(name="y", parents=("x",), dist=lambda x: dist.Normal(x, 1.0))
    )
    result = infer(diag, ["x"], observed={"y": 2.0}, num_samples=300)
    assert len(result["x"].values) == 300
    assert result["x"].mean() == pytest.approx(1.0, abs=0.2)


def test_rng_key_controls_the_draws():
    diagram = _continuous_diagram()
    default = infer(diagram, ["x"])["x"].values
    same = infer(diagram, ["x"], rng_key=jax.random.PRNGKey(0))["x"].values
    other = infer(diagram, ["x"], rng_key=jax.random.PRNGKey(1))["x"].values
    assert jnp.array_equal(default, same)
    assert not jnp.array_equal(default, other)


# --- policy binding -----------------------------------------------------------


def test_policy_collapses_id_to_bn():
    """
    A fully bound ID behaves like a BN with the decisions as evidence.

    Exact posterior: P(rain=yes | umbrella=1, wet=1) = 0.2 * 0.95 / 0.43
    ≈ 0.442.
    """
    result = infer(
        _umbrella_diagram(),
        ["rain"],
        observed={"wet": 1},
        policy={"umbrella": 1},
    )
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].marginal() == pytest.approx([0.558, 0.442], abs=0.04)


def test_policy_on_continuous_diagram():
    """A continuous chance node plus a bound decision still infers."""
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d", states=("no", "yes")))
    diag.add_node(
        ChanceNode(
            name="x", parents=("d",), dist=lambda d: dist.Normal(loc=2.0 * d, scale=1.0)
        )
    )
    diag.add_node(
        UtilityNode(
            name="u", parents=("x", "d"), values=lambda x, d: float(x) * float(d)
        )
    )
    result = infer(diag, ["x"], policy={"d": 1}, num_samples=2000)
    assert isinstance(result["x"], Posterior)
    assert result["x"].states is None
    assert len(result["x"].values) == 2000
    assert result["x"].mean() == pytest.approx(2.0, abs=0.2)


def test_utility_node_alone_does_not_block_infer():
    """Utilities are ignored; a chance+utility diagram infers fine."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
        )
    )
    diag.add_node(
        UtilityNode(
            name="dryness",
            parents=("rain",),
            values=lambda rain: float(rain),
        )
    )
    result = infer(diag, ["rain"])
    assert isinstance(result["rain"], Posterior)
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_partially_bound_decisions_raise():
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d1", states=("no", "yes")))
    diag.add_node(DecisionNode(name="d2", states=("no", "yes")))
    diag.add_node(
        ChanceNode(
            name="x",
            parents=("d1", "d2"),
            dist=lambda d1, d2: dist.Normal(loc=d1 + d2, scale=1.0),
        )
    )
    with pytest.raises(InferenceError, match="unbound decision.*d2"):
        infer(diag, ["x"], policy={"d1": 0})


def test_decision_off_the_query_path_needs_no_policy():
    """``umbrella`` only feeds a utility, so it cannot change ``rain``."""
    result = infer(_decision_diagram(), ["rain"])
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_decision_downstream_of_query_needs_no_policy():
    """``umbrella`` is a child of ``rain``; with ``wet`` unobserved it is irrelevant."""
    result = infer(_umbrella_diagram(), ["rain"])
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_policy_for_an_irrelevant_decision_is_ignored():
    result = infer(_decision_diagram(), ["rain"], policy={"umbrella": 1})
    assert result["rain"].marginal() == pytest.approx([0.8, 0.2], abs=0.04)


def test_irrelevant_unobserved_chance_node_is_ignored():
    """
    A continuous node off the query path does not force NUTS.

    ``noise`` is a ``Delta``, which NUTS rejects; it only runs if the node
    takes part in inference.
    """
    diag = _rain_wet_grass()
    diag.add_node(
        ChanceNode(
            name="noise", parents=("rain",), dist=lambda rain: dist.Delta(1.0 * rain)
        )
    )
    result = infer(diag, ["rain"], observed={"wet_grass": 1})
    assert result["rain"].marginal() == pytest.approx([0.457, 0.543], abs=0.04)


def test_policy_without_evidence_samples_forward():
    """
    Bound decisions alone are not evidence: the draws come from forward sampling.

    ``gain`` is a deterministic function of the decision and a continuous
    chance node, so NUTS would reject it.
    """
    diag = InfluenceDiagram()
    diag.add_node(DecisionNode(name="d", states=("low", "high")))
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(10.0, 2.0)))
    diag.add_node(
        ChanceNode(
            name="gain",
            parents=("d", "x"),
            dist=lambda d, x: dist.Delta(jnp.where(d == 1, 2.0, 1.0) * x),
        )
    )
    result = infer(diag, ["gain"], policy={"d": 1})
    assert result["gain"].mean() == pytest.approx(20.0, abs=0.4)
    assert result["gain"].std() == pytest.approx(4.0, abs=0.3)


def test_policy_unknown_name_raises():
    with pytest.raises(InferenceError, match="unknown names.*bogus"):
        infer(_umbrella_diagram(), ["rain"], policy={"bogus": 0})


def test_policy_on_bayesian_network_raises():
    with pytest.raises(InferenceError, match="decision nodes only"):
        infer(_rain_wet_grass(), ["rain"], policy={"rain": 0})


def test_querying_a_decision_raises():
    with pytest.raises(InferenceError, match="chance and utility nodes only"):
        infer(_umbrella_diagram(), ["umbrella"], policy={"umbrella": 1})


def test_observing_a_decision_raises():
    with pytest.raises(InferenceError, match="observed= names chance nodes only"):
        infer(_umbrella_diagram(), ["wet"], observed={"umbrella": 1})


def test_observing_a_utility_raises():
    with pytest.raises(InferenceError, match="observed= names chance nodes only"):
        infer(_umbrella_diagram(), ["rain"], observed={"dryness": 1.0})


# --- utility queries ----------------------------------------------------------


STOCK_LEVELS = jnp.array([80.0, 120.0])


def _shop_diagram() -> InfluenceDiagram:
    """
    ``demand ~ Normal(100, 20)``, a stock decision, and ``profit``.

    A noisy ``report`` of the demand can be observed, which puts a
    continuous latent under NUTS.
    """
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="demand", dist=lambda: dist.Normal(100.0, 20.0)))
    diag.add_node(
        ChanceNode(
            name="report",
            parents=("demand",),
            dist=lambda demand: dist.Normal(demand, 5.0),
        )
    )
    diag.add_node(DecisionNode(name="stock", states=("80", "120")))
    diag.add_node(
        UtilityNode(
            name="profit",
            parents=("stock", "demand"),
            values=lambda stock, demand: (
                10.0 * jnp.minimum(STOCK_LEVELS[stock], demand)
                - 2.0 * STOCK_LEVELS[stock]
            ),
        )
    )
    return diag


def _guess_diagram() -> InfluenceDiagram:
    """``rain -> wet``, a guess, and ``hit`` = 1 when the guess matches the rain."""
    diag = _rain_wet_grass()
    diag.add_node(DecisionNode(name="guess", states=("no", "yes")))
    diag.add_node(
        UtilityNode(
            name="hit",
            parents=("rain", "guess"),
            values=lambda rain, guess: jnp.where(rain == guess, 1.0, 0.0),
        )
    )
    return diag


def test_utility_query_returns_a_continuous_posterior():
    result = infer(_shop_diagram(), ["profit"], policy={"stock": 0}, num_samples=2000)
    assert result["profit"].states is None
    assert len(result["profit"].values) == 2000


def test_utility_query_under_a_policy():
    """
    Stocking 80 sells out unless demand drops below 80, which is P ≈ 0.16.

    With ``demand ~ N(100, 20)`` and ``profit = 10·min(80, demand) − 160``,
    profit is capped at 640.
    """
    profit = infer(_shop_diagram(), ["profit"], policy={"stock": 0})["profit"]
    assert profit.values.max() == pytest.approx(640.0)
    capped = jnp.isclose(profit.values, 640.0).mean()
    assert capped == pytest.approx(0.84, abs=0.03)


def test_utility_query_with_a_continuous_latent():
    """A high report pins demand above both stock levels, so profit is capped."""
    profit = infer(
        _shop_diagram(), ["profit"], policy={"stock": 1}, observed={"report": 200.0}
    )["profit"]
    assert profit.mean() == pytest.approx(10.0 * 120 - 2.0 * 120, abs=1.0)


def test_utility_query_with_discrete_enumeration():
    """``hit`` is 1 exactly when it rains, so its mean is P(rain=yes | wet) ≈ 0.543."""
    result = infer(
        _guess_diagram(),
        ["hit", "rain"],
        observed={"wet_grass": 1},
        policy={"guess": 1},
    )
    assert result["hit"].mean() == pytest.approx(0.543, abs=0.04)


def test_utility_draws_line_up_with_chance_draws():
    result = infer(
        _guess_diagram(),
        ["hit", "rain"],
        observed={"wet_grass": 1},
        policy={"guess": 1},
    )
    assert jnp.array_equal(result["hit"].values, result["rain"].values)


def test_utility_with_an_observed_parent():
    diag = _rain_wet_grass()
    diag.add_node(
        UtilityNode(
            name="cleanup",
            parents=("wet_grass",),
            values=lambda wet_grass: -5.0 * wet_grass,
        )
    )
    result = infer(diag, ["cleanup"], observed={"wet_grass": 1})
    assert jnp.all(result["cleanup"].values == -5)


def test_utility_query_needs_its_upstream_decisions_bound():
    with pytest.raises(InferenceError, match="unbound decision.*stock"):
        infer(_shop_diagram(), ["profit"])


def test_utilities_do_not_change_other_posteriors():
    with_utility = infer(
        _guess_diagram(),
        ["rain", "hit"],
        observed={"wet_grass": 1},
        policy={"guess": 0},
    )
    without = infer(_guess_diagram(), ["rain"], observed={"wet_grass": 1})
    assert jnp.array_equal(with_utility["rain"].values, without["rain"].values)


@pytest.mark.parametrize("domain", [Real(), Interval(0.0, 100.0)])
@pytest.mark.parametrize("rule", [jnp.array(0.5), lambda: 0.5])
def test_infer_evaluates_supplied_continuous_actions(domain, rule):
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=domain))
    diagram.add_node(
        UtilityNode("payoff", parents=("power",), values=lambda power: -(power**2))
    )
    result = infer(diagram, ["payoff"], policy={"power": rule}, num_samples=2)
    assert jnp.all(result["payoff"].values == -0.25)


def test_inference_backend_evaluates_continuous_fixed_decisions():
    diagram = InfluenceDiagram()
    diagram.add_node(DecisionNode("power", action_space=Real()))
    result = samples(diagram.snapshot(), observed={"power": 0.5}, num_samples=2)
    assert jnp.all(result["power"] == 0.5)


def test_infer_prunes_irrelevant_continuous_decisions():
    diagram = _continuous_diagram()
    expected = infer(diagram, ["x"], num_samples=32)["x"].values
    diagram.add_node(DecisionNode("power", action_space=Interval(0.0, 100.0)))
    result = infer(diagram, ["x"], policy={"power": 0}, num_samples=32)
    assert jnp.array_equal(result["x"].values, expected)
