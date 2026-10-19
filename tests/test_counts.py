"""Numerical counts in forward simulation and observed-count likelihoods."""

from math import comb, exp, sqrt

import jax
import jax.numpy as jnp
import numpyro
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
from pylimid.numpyro import samples, to_model
from pylimid.solver.backward_induction import solve as backward_induction_solve
from pylimid.solver.scan.batched import solve as batched_solve
from pylimid.solver.scan.intervention import solve as intervention_solve


@pytest.mark.parametrize(
    "distribution,mean,variance",
    [
        (lambda: dist.Poisson(3.0), 3.0, 3.0),
        (lambda: dist.Binomial(total_count=10, probs=0.2), 2.0, 1.6),
        (lambda: dist.Binomial(total_count=10, logits=0.0), 5.0, 2.5),
        (lambda: dist.NegativeBinomial2(mean=3.0, concentration=2.0), 3.0, 7.5),
        (lambda: dist.Geometric(probs=0.5), 1.0, 2.0),
    ],
    ids=[
        "poisson",
        "binomial-probs",
        "binomial-logits",
        "negative-binomial",
        "geometric",
    ],
)
def test_forward_counts_keep_numerical_values_and_summaries(
    distribution, mean, variance
):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=distribution))

    posterior = infer(diagram, ["count"], num_samples=4000)["count"]

    assert not posterior.is_categorical
    assert posterior.values.shape == (4000,)
    assert jnp.all(posterior.values >= 0)
    assert jnp.all(posterior.values == jnp.floor(posterior.values))
    assert posterior.mean() == pytest.approx(mean, abs=0.15)
    assert posterior.std() == pytest.approx(sqrt(variance), abs=0.2)
    lower, upper = posterior.hdi()
    assert lower <= posterior.mean() <= upper
    with pytest.raises(ValueError, match="requires a categorical"):
        posterior.marginal()


@pytest.mark.parametrize("family", ["poisson", "binomial"])
@pytest.mark.parametrize(
    "execution", ["inference", "backward_induction", "batched", "intervention"]
)
def test_counts_with_chance_and_decision_parents_feed_utilities(family, execution):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="weather", states=("dry", "wet"), dist=lambda: dist.Bernoulli(0.5)
        )
    )
    diagram.add_node(
        DecisionNode(name="action", parents=("weather",), states=("no", "yes"))
    )

    def distribution(weather, action):
        if family == "poisson":
            return dist.Poisson(2.0 + 4 * weather + action)
        return dist.Binomial(
            total_count=jnp.array([5, 10])[weather], probs=0.2 + 0.1 * action
        )

    expected_utility = 5.5 if family == "poisson" else 8.0
    diagram.add_node(
        ChanceNode(name="count", parents=("weather", "action"), dist=distribution)
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("weather", "action", "count"),
            values=lambda weather, action, count: 10.0 * (weather == action) - count,
        )
    )
    policy = {"action": {(0,): 0, (1,): 1}}

    if execution == "inference":
        result = infer(diagram, ["count", "payoff"], policy=policy, num_samples=1500)
        assert jnp.array_equal(result["payoff"].values, 10 - result["count"].values)
        assert result["payoff"].mean() == pytest.approx(expected_utility, abs=0.4)
    else:
        solver = {
            "backward_induction": backward_induction_solve,
            "batched": batched_solve,
            "intervention": intervention_solve,
        }[execution]
        result = solver(diagram.snapshot(), num_samples=1500)
        assert result.policy == policy
        assert result.expected_utility == pytest.approx(expected_utility, abs=0.4)


def test_observed_poisson_count_updates_continuous_rate():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="rate", dist=lambda: dist.Gamma(1.0, 1.0)))
    diagram.add_node(
        ChanceNode(
            name="count", parents=("rate",), dist=lambda rate: dist.Poisson(rate)
        )
    )

    result = infer(diagram, ["rate", "count"], observed={"count": 4}, num_samples=2000)

    # Gamma(1, 1) with one Poisson observation of four gives Gamma(5, 2).
    assert result["rate"].mean() == pytest.approx(2.5, abs=0.15)
    assert result["rate"].std() == pytest.approx(sqrt(5) / 2, abs=0.15)
    assert jnp.all(result["count"].values == 4)
    assert result["count"].mean() == 4


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_observed_counts_update_categorical_parent(family):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="batch", states=("small", "large"), dist=lambda: dist.Bernoulli(0.5)
        )
    )

    def distribution(batch):
        if family == "poisson":
            return dist.Poisson(3.0 + 3 * batch)
        return dist.Binomial(total_count=jnp.array([5, 10])[batch], probs=0.2)

    if family == "poisson":
        likelihood_ratio = 16 * exp(-3)
        observed = 4
    else:
        likelihood_ratio = comb(10, 3) / comb(5, 3) * 0.8**5
        observed = 3
    diagram.add_node(ChanceNode(name="count", parents=("batch",), dist=distribution))

    result = infer(
        diagram, ["batch", "count"], observed={"count": observed}, num_samples=4000
    )

    expected = likelihood_ratio / (1 + likelihood_ratio)
    assert result["batch"].marginal()[1] == pytest.approx(expected, abs=0.03)
    assert jnp.all(result["count"].values == observed)


@pytest.mark.parametrize("family", ["poisson", "binomial"])
@pytest.mark.parametrize("continuous_parent", [False, True])
def test_latent_counts_are_rejected_even_when_not_queried(family, continuous_parent):
    diagram = InfluenceDiagram()

    def distribution(rate):
        if family == "poisson":
            return dist.Poisson(rate)
        return dist.Binomial(total_count=10, probs=rate / (1 + rate))

    if continuous_parent:
        diagram.add_node(ChanceNode(name="rate", dist=lambda: dist.Gamma(1.0, 1.0)))
        count = ChanceNode(name="count", parents=("rate",), dist=distribution)
    else:
        count = ChanceNode(name="count", dist=lambda: distribution(3.0))
    diagram.add_node(count)
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    with pytest.raises(InferenceError, match="unobserved numerical discrete.*count"):
        infer(diagram, ["report"], observed={"report": 2.0}, num_samples=10)


def test_all_observed_counts_need_no_latent_sampler():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=lambda: dist.Poisson(3.0)))

    result = infer(diagram, ["count"], observed={"count": 4}, num_samples=20)

    assert jnp.array_equal(result["count"].values, jnp.full(20, 4))


def test_unrelated_latent_count_is_pruned_from_posterior():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="unused_count", dist=lambda: dist.Poisson(3.0)))
    diagram.add_node(
        ChanceNode(
            name="weather", states=("dry", "wet"), dist=lambda: dist.Bernoulli(0.5)
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("weather",),
            states=("no", "yes"),
            dist=lambda weather: dist.Bernoulli(0.2 + 0.6 * weather),
        )
    )

    result = infer(diagram, ["weather"], observed={"report": 1}, num_samples=2000)

    assert result["weather"].marginal()[1] == pytest.approx(0.8, abs=0.03)


@pytest.mark.parametrize(
    "execution", ["inference", "backward_induction", "batched", "intervention"]
)
def test_counts_cannot_index_tabular_policies(execution):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=lambda: dist.Poisson(3.0)))
    diagram.add_node(
        DecisionNode(name="action", parents=("count",), states=("no", "yes"))
    )
    diagram.add_node(
        UtilityNode(
            name="payoff", parents=("action",), values=lambda action: 1.0 * action
        )
    )

    with pytest.raises(ValueError, match="action.*count.*without named states"):
        if execution == "inference":
            infer(diagram, ["payoff"], policy={"action": {(0,): 0}}, num_samples=10)
        else:
            solver = {
                "backward_induction": backward_induction_solve,
                "batched": batched_solve,
                "intervention": intervention_solve,
            }[execution]
            solver(diagram.snapshot(), num_samples=10)


def test_vector_discrete_events_are_rejected():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="counts",
            dist=lambda: dist.Multinomial(total_count=3, probs=jnp.array([0.4, 0.6])),
        )
    )

    with pytest.raises(ValueError, match="counts.*scalar events"):
        infer(diagram, ["counts"], num_samples=10)


def test_batched_scalar_counts_keep_component_summaries():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="counts", dist=lambda: dist.Poisson(jnp.array([2.0, 5.0])))
    )

    posterior = infer(diagram, ["counts"], num_samples=2000)["counts"]

    assert posterior.values.shape == (2000, 2)
    assert posterior.mean() == pytest.approx([2.0, 5.0], abs=0.15)


def test_numerical_counts_are_not_marked_for_enumeration():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="count", dist=lambda: dist.Binomial(total_count=10, probs=0.2))
    )
    model = to_model(diagram.snapshot())

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace()

    assert "enumerate" not in trace["count"]["infer"]


def test_low_level_samples_reject_latent_counts_with_evidence():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=lambda: dist.Poisson(3.0)))
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    with pytest.raises(InferenceError, match="unobserved numerical discrete.*count"):
        samples(diagram.snapshot(), observed={"report": 2.0}, num_samples=10)
