"""Forward sampling and discrete, continuous, and mixed posterior draws."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid._numpyro.samplers import samples
from pylimid.errors import InferenceError
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode
from tests._numpyro.diagrams import rain_wet_grass_diagram


def test_samples_prior_returns_named_sites() -> None:
    """samples() with no observations returns one array per node."""
    draws = samples(rain_wet_grass_diagram().snapshot(), num_samples=1000)
    assert set(draws.keys()) == {"rain", "wet_grass"}
    assert draws["rain"].shape == (1000,)
    assert draws["wet_grass"].shape == (1000,)


def test_samples_prior_marginal_matches_handcomputed() -> None:
    """
    Empirical marginals match the hand-computed values.

    rain=yes has prior 0.2. P(wet_grass=wet) marginalizes over rain:

        P(wet) = P(wet|no)*P(no) + P(wet|yes)*P(yes)
               = 0.20*0.8 + 0.95*0.2 = 0.35

    Forward sampling must reproduce both within sampling tolerance.
    """
    draws = samples(rain_wet_grass_diagram().snapshot(), num_samples=20_000)

    # rain: index 1 == 'yes'. Prior P(yes) = 0.2.
    p_rain_yes = float(jnp.mean(draws["rain"] == 1))
    # wet_grass: index 1 == 'wet'. Hand-computed P(wet) = 0.35.
    p_wet = float(jnp.mean(draws["wet_grass"] == 1))

    assert p_rain_yes == pytest.approx(0.2, abs=0.02)
    assert p_wet == pytest.approx(0.35, abs=0.02)


def test_samples_prior_continuous_root() -> None:
    """A continuous root node samples to plausible support, not indices."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="height",
            dist=lambda: dist.Normal(loc=170.0, scale=5.0),
        )
    )
    draws = samples(diag.snapshot(), num_samples=20_000)
    heights = draws["height"]
    assert heights.shape == (20_000,)
    assert float(jnp.mean(heights)) == pytest.approx(170.0, abs=0.5)
    assert float(jnp.std(heights)) == pytest.approx(5.0, abs=0.5)


def test_samples_query_subset() -> None:
    """query= returns only the requested nodes."""
    draws = samples(
        rain_wet_grass_diagram().snapshot(),
        query=["rain"],
        num_samples=100,
    )
    assert set(draws.keys()) == {"rain"}


def test_samples_rng_key_is_deterministic() -> None:
    """The same rng_key reproduces the same draws."""
    snapshot = rain_wet_grass_diagram().snapshot()
    a = samples(snapshot, num_samples=100, rng_key=jax.random.PRNGKey(7))
    b = samples(snapshot, num_samples=100, rng_key=jax.random.PRNGKey(7))
    assert set(a.keys()) == set(b.keys())
    for name in a:
        assert jnp.array_equal(a[name], b[name])


def test_samples_posterior_discrete_with_observation() -> None:
    """
    Enumeration recovers the exact posterior on rain given wet_grass=wet.

    Hand-computed:
        P(rain=yes | wet) = P(wet|yes)*P(yes) / P(wet)
                          = 0.95 * 0.2 / 0.35 ≈ 0.543
    """
    draws = samples(
        rain_wet_grass_diagram().snapshot(),
        observed={"wet_grass": 1},
        num_samples=2000,
    )
    p_rain_yes = float(jnp.mean(draws["rain"] == 1))
    assert p_rain_yes == pytest.approx(0.543, abs=0.03)


def test_samples_returns_observed_sites_clamped() -> None:
    """Observed nodes appear in the draws as their clamped value."""
    draws = samples(
        rain_wet_grass_diagram().snapshot(),
        observed={"wet_grass": 1},
        num_samples=200,
    )
    assert set(draws.keys()) == {"rain", "wet_grass"}
    assert draws["wet_grass"].shape == (200,)
    assert bool(jnp.all(draws["wet_grass"] == 1))


def test_samples_posterior_mixed_discrete_continuous() -> None:
    """
    Mixed BN: rain (discrete) -> height (continuous).

    Observe height=175; infer P(rain=yes | height=175).

    height | rain=no  ~ Normal(170, 5)
    height | rain=yes ~ Normal(175, 5)

    P(rain=yes | height=175) ∝ N(175|175,5) * 0.2
    P(rain=no  | height=175) ∝ N(175|170,5) * 0.8

    N(175|175,5) = N(0|0,5) ≈ 0.0798
    N(175|170,5) = N(5|0,5) ≈ 0.6065 * 0.0798 ≈ 0.0484

    P(yes|175) ≈ 0.0798*0.2 / (0.0798*0.2 + 0.0484*0.8) ≈ 0.292

    Every latent node is discrete, so this takes the exact enumeration path
    (the observed continuous site is clamped).
    """
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
            name="height",
            parents=("rain",),
            dist=lambda rain: dist.Normal(
                loc=jnp.where(rain == 0, 170.0, 175.0), scale=5.0
            ),
        )
    )
    draws = samples(
        diag.snapshot(),
        observed={"height": jnp.array(175.0)},
        num_samples=2000,
    )
    p_rain_yes = float(jnp.mean(draws["rain"] == 1))
    assert p_rain_yes == pytest.approx(0.292, abs=0.03)


def test_samples_posterior_continuous_hierarchical() -> None:
    """
    MCMC recovers the mu posterior given a height observation.

    Model: mu ~ Normal(0, 10); height ~ Normal(mu, 5). Observe height=175.
    Posterior: mu ~ Normal(175 * 100 / (100+25), sqrt(1/(1/100 + 1/25)))
             = Normal(140, ~4.47)
    """
    draws = samples(
        _continuous_hierarchical_diagram().snapshot(),
        observed={"height": jnp.array(175.0)},
        num_samples=2000,
    )
    mu_samples = draws["mu"]
    assert mu_samples.shape == (2000,)
    # Normal-Normal conjugate: posterior mean = 140, posterior std ≈ 4.47
    assert float(jnp.mean(mu_samples)) == pytest.approx(140.0, abs=1.0)
    assert float(jnp.std(mu_samples)) == pytest.approx(4.47, abs=0.5)


def test_samples_posterior_mcmc_resamples_discrete() -> None:
    """
    The MCMC path resamples discrete latents and covers every node.

    rain (discrete) -> bias (continuous) -> height (observed). NUTS samples
    ``bias`` (the only continuous latent); the conditional pass resamples
    ``rain`` given each bias draw. The returned dict covers all three nodes.

    Observing height=175 favors bias≈0 and therefore rain=no (bias mean 0)
    over rain=yes (bias mean 2).
    """
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="bias",
            parents=("rain",),
            dist=lambda rain: dist.Normal(
                loc=jnp.where(rain == 0, 0.0, 2.0), scale=1.0
            ),
        )
    )
    diag.add_node(
        ChanceNode(
            name="height",
            parents=("bias",),
            dist=lambda bias: dist.Normal(loc=175.0 + bias, scale=1.0),
        )
    )
    draws = samples(
        diag.snapshot(),
        observed={"height": jnp.array(175.0)},
        num_samples=500,
        num_warmup=300,
    )
    assert set(draws.keys()) == {"rain", "bias", "height"}
    assert draws["rain"].shape == (500,)
    assert draws["bias"].shape == (500,)
    assert bool(jnp.all(draws["height"] == 175.0))
    assert float(jnp.mean(draws["rain"] == 1)) < 0.5


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


def _continuous_hierarchical_diagram() -> InfluenceDiagram:
    """Mu ~ Normal(0, 10); height ~ Normal(mu, 5)."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="mu",
            dist=lambda: dist.Normal(loc=0.0, scale=10.0),
        )
    )
    diag.add_node(
        ChanceNode(
            name="height",
            parents=("mu",),
            dist=lambda mu: dist.Normal(loc=mu, scale=5.0),
        )
    )
    return diag
