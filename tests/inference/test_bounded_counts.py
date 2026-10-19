"""Finite numerical count posteriors checked against explicit probability sums."""

from math import comb, exp, lgamma, sqrt

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import ChanceNode, InferenceError, InfluenceDiagram, UtilityNode, infer


def _binomial_masses(trials, probability):
    return jnp.array(
        [
            comb(trials, count)
            * probability**count
            * (1 - probability) ** (trials - count)
            for count in range(trials + 1)
        ]
    )


def _beta_binomial_masses(trials, alpha, beta):
    return jnp.array(
        [
            comb(trials, count)
            * exp(
                lgamma(alpha + count)
                + lgamma(beta + trials - count)
                - lgamma(alpha + beta + trials)
                - lgamma(alpha)
                - lgamma(beta)
                + lgamma(alpha + beta)
            )
            for count in range(trials + 1)
        ]
    )


@pytest.mark.parametrize("logits", [False, True], ids=["probs", "logits"])
def test_binomial_posterior_matches_exact_probabilities_and_moments(logits):
    trials, probability = 10, 0.2
    diagram = InfluenceDiagram()

    def distribution():
        if logits:
            return dist.Binomial(
                total_count=trials, logits=jnp.log(probability / (1 - probability))
            )
        return dist.Binomial(total_count=trials, probs=probability)

    diagram.add_node(ChanceNode(name="defects", dist=distribution))
    diagram.add_node(
        ChanceNode(
            name="alarm",
            parents=("defects",),
            states=("off", "on"),
            dist=lambda defects: dist.Bernoulli(0.05 + 0.08 * defects),
        )
    )
    diagram.add_node(
        UtilityNode(
            name="cost", parents=("defects",), values=lambda defects: 2.0 * defects
        )
    )

    result = infer(
        diagram, ["defects", "cost"], observed={"alarm": 1}, num_samples=6000
    )

    support = jnp.arange(trials + 1)
    weights = _binomial_masses(trials, probability) * (0.05 + 0.08 * support)
    expected = weights / weights.sum()
    mean = jnp.sum(support * expected)
    variance = jnp.sum((support - mean) ** 2 * expected)
    defects = result["defects"]
    assert not defects.is_categorical
    assert defects.mean() == pytest.approx(float(mean), abs=0.06)
    assert defects.std() == pytest.approx(sqrt(float(variance)), abs=0.06)
    assert jnp.bincount(defects.values, length=trials + 1) / 6000 == pytest.approx(
        expected, abs=0.025
    )
    assert jnp.array_equal(result["cost"].values, 2 * defects.values)
    with pytest.raises(ValueError, match="requires a categorical"):
        defects.marginal()


@pytest.mark.parametrize(
    "trials,probability,expected", [(0, 0.2, 0), (5, 0.0, 0), (5, 1.0, 5)]
)
def test_degenerate_binomial_posteriors_preserve_counts(trials, probability, expected):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="count",
            dist=lambda: dist.Binomial(total_count=trials, probs=probability),
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    result = infer(diagram, ["count"], observed={"report": expected}, num_samples=100)

    assert jnp.all(result["count"].values == expected)


@pytest.mark.parametrize("continuous", [False, True], ids=["categorical", "continuous"])
def test_zero_trial_count_preserves_parent_posterior(continuous):
    diagram = InfluenceDiagram()
    if continuous:
        parent = ChanceNode(name="probability", dist=lambda: dist.Beta(2.0, 3.0))
    else:
        parent = ChanceNode(
            name="probability", states=("low", "high"), dist=lambda: dist.Bernoulli(0.5)
        )
    diagram.add_node(parent)
    diagram.add_node(
        ChanceNode(
            name="count",
            parents=("probability",),
            dist=lambda probability: dist.Binomial(
                total_count=0, probs=0.1 + 0.5 * probability
            ),
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    result = infer(
        diagram, ["probability", "count"], observed={"report": 0.0}, num_samples=2000
    )

    assert jnp.all(result["count"].values == 0)
    if continuous:
        assert result["probability"].mean() == pytest.approx(0.4, abs=0.03)
    else:
        assert result["probability"].marginal() == pytest.approx([0.5, 0.5], abs=0.03)


def test_multiple_finite_counts_preserve_exact_joint_posterior():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="first", dist=lambda: dist.Binomial(total_count=4, probs=0.2))
    )
    diagram.add_node(
        ChanceNode(name="second", dist=lambda: dist.Binomial(total_count=3, probs=0.4))
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("first", "second"),
            dist=lambda first, second: dist.Normal(first + second, 1.0),
        )
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("first", "second"),
            values=lambda first, second: 2.0 * first - second,
        )
    )

    result = infer(
        diagram,
        ["first", "second", "payoff"],
        observed={"report": 2.0},
        num_samples=6000,
    )

    first = jnp.arange(5)[:, None]
    second = jnp.arange(4)[None, :]
    weights = (
        _binomial_masses(4, 0.2)[:, None]
        * _binomial_masses(3, 0.4)[None, :]
        * jnp.exp(-0.5 * (2 - first - second) ** 2)
    )
    expected = weights / weights.sum()
    indices = 4 * result["first"].values + result["second"].values
    actual = jnp.bincount(indices, length=20).reshape((5, 4)) / 6000
    assert actual == pytest.approx(expected, abs=0.025)
    assert jnp.array_equal(
        result["payoff"].values, 2 * result["first"].values - result["second"].values
    )


@pytest.mark.parametrize(
    "query_count", [False, True], ids=["hidden-count", "queried-count"]
)
def test_fixed_bound_count_updates_categorical_parent_and_preserves_joint_draws(
    query_count,
):
    trials = 5
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="supplier", states=("good", "bad"), dist=lambda: dist.Bernoulli(0.5)
        )
    )
    diagram.add_node(
        ChanceNode(
            name="count",
            parents=("supplier",),
            dist=lambda supplier: dist.Binomial(
                total_count=trials, probs=0.1 + 0.5 * supplier
            ),
        )
    )
    diagram.add_node(
        ChanceNode(
            name="alarm",
            parents=("count",),
            states=("off", "on"),
            dist=lambda count: dist.Bernoulli(0.1 + 0.12 * count),
        )
    )

    query = ["supplier", "count"] if query_count else ["supplier"]
    result = infer(diagram, query, observed={"alarm": 1}, num_samples=6000)

    support = jnp.arange(trials + 1)
    masses = jnp.stack((_binomial_masses(trials, 0.1), _binomial_masses(trials, 0.6)))
    weights = 0.5 * masses * (0.1 + 0.12 * support)
    expected_joint = weights / weights.sum()
    assert result["supplier"].marginal() == pytest.approx(
        expected_joint.sum(axis=1), abs=0.025
    )
    if query_count:
        joint_indices = (
            result["supplier"].values * (trials + 1) + result["count"].values
        )
        actual_joint = (
            jnp.bincount(joint_indices, length=2 * (trials + 1)).reshape(
                (2, trials + 1)
            )
            / 6000
        )
        assert actual_joint == pytest.approx(expected_joint, abs=0.025)


@pytest.mark.parametrize("logits", [False, True], ids=["probs", "logits"])
def test_mixed_beta_binomial_posterior_matches_exact_mixture(logits):
    trials, alpha, beta = 5, 2.0, 3.0
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="probability", dist=lambda: dist.Beta(alpha, beta))
    )

    def distribution(probability):
        if logits:
            return dist.Binomial(
                total_count=trials,
                logits=jnp.log(probability) - jnp.log1p(-probability),
            )
        return dist.Binomial(total_count=trials, probs=probability)

    diagram.add_node(
        ChanceNode(name="count", parents=("probability",), dist=distribution)
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("count", "probability"),
            values=lambda count, probability: count * probability,
        )
    )

    result = infer(
        diagram,
        ["count", "probability", "payoff"],
        observed={"report": 2.0},
        num_samples=3000,
    )

    support = jnp.arange(trials + 1)
    weights = _beta_binomial_masses(trials, alpha, beta) * jnp.exp(
        -0.5 * (2 - support) ** 2
    )
    weights = weights / weights.sum()
    conditional_means = (alpha + support) / (alpha + beta + trials)
    assert result["count"].mean() == pytest.approx(
        float(jnp.sum(support * weights)), abs=0.08
    )
    assert result["probability"].mean() == pytest.approx(
        float(jnp.sum(conditional_means * weights)), abs=0.025
    )
    assert result["payoff"].mean() == pytest.approx(
        float(jnp.sum(support * conditional_means * weights)), abs=0.05
    )
    assert jnp.array_equal(
        result["payoff"].values, result["count"].values * result["probability"].values
    )


def test_beta_binomial_count_posterior_matches_exact_masses():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="count", dist=lambda: dist.BetaBinomial(2.0, 3.0, total_count=5)
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    posterior = infer(diagram, ["count"], observed={"report": 2.0}, num_samples=5000)[
        "count"
    ]

    support = jnp.arange(6)
    weights = _beta_binomial_masses(5, 2.0, 3.0) * jnp.exp(-0.5 * (2 - support) ** 2)
    expected = weights / weights.sum()
    assert jnp.bincount(posterior.values, length=6) / 5000 == pytest.approx(
        expected, abs=0.025
    )


def test_shifted_finite_ranges_are_rejected_instead_of_returning_indices():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=lambda: dist.DiscreteUniform(3, 5)))
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    with pytest.raises(InferenceError, match="fixed zero-based integer range.*count"):
        infer(diagram, ["count"], observed={"report": 4.0}, num_samples=10)


@pytest.mark.parametrize(
    "distribution",
    [
        lambda: dist.Binomial(total_count=10, probs=jnp.array([0.2, 0.4])),
        lambda: dist.DiscreteUniform(jnp.array([0, 0]), 5),
    ],
    ids=["binomial", "uniform"],
)
def test_batched_count_posteriors_are_rejected(distribution):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="counts",
            dist=distribution,
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("counts",),
            dist=lambda counts: dist.Normal(jnp.sum(counts), 1.0),
        )
    )

    with pytest.raises(
        InferenceError, match="counts.*scalar distribution.*batch shape"
    ):
        infer(diagram, ["counts"], observed={"report": 4.0}, num_samples=10)


def test_parent_dependent_trial_counts_still_sample_forward_and_fail_enumeration():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="batch", states=("small", "large"), dist=lambda: dist.Bernoulli(0.5)
        )
    )
    diagram.add_node(
        ChanceNode(
            name="count",
            parents=("batch",),
            dist=lambda batch: dist.Binomial(
                total_count=jnp.array([5, 10])[batch], probs=0.2
            ),
        )
    )
    diagram.add_node(
        ChanceNode(
            name="report",
            parents=("count",),
            dist=lambda count: dist.Normal(count, 1.0),
        )
    )

    result = infer(diagram, ["batch", "count"], num_samples=2000)
    assert jnp.all(result["count"].values <= jnp.array([5, 10])[result["batch"].values])
    assert result["count"].mean() == pytest.approx(1.5, abs=0.15)
    with pytest.raises((jax.errors.ConcretizationTypeError, NotImplementedError)):
        infer(diagram, ["count"], observed={"report": 2.0}, num_samples=10)
