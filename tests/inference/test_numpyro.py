"""Tests for :mod:`decisionpy.inference.numpyro` and
:mod:`decisionpy.inference._numpyro_model`."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.contrib.funsor import infer_discrete
from numpyro.infer import MCMC, NUTS, Predictive

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.inference.numpyro.model import to_model

# Skip the whole module when the numpyro extra is not installed. The heavy
# imports above already fail in that case, so the guard is belt-and-braces.
numpyro_missing: bool = False
try:
    import numpyro  # noqa: F401
except ImportError:  # pragma: no cover
    numpyro_missing = True
pytestmark = pytest.mark.skipif(numpyro_missing, reason="numpyro extra not installed")


# --- helpers ----------------------------------------------------------------

#: Hand-checked BN used across tests: rain ~ Bernoulli(0.2), then
#: wet_grass | rain. ``_P_WET[rain]`` is the CPT row for ``wet_grass``.
_PRIOR_RAIN = jnp.array([0.8, 0.2])  # P(rain=no)=0.8, P(rain=yes)=0.2
_P_WET = jnp.array(
    [
        [0.80, 0.20],  # rain=no  -> P(dry)=0.8, P(wet)=0.2
        [0.05, 0.95],  # rain=yes -> P(dry)=0.05, P(wet)=0.95
    ]
)


def _rain_node() -> ChanceNode:
    """Root node: rain ~ Categorical([0.8, 0.2])."""
    return ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=_PRIOR_RAIN),
    )


def _wet_grass_node() -> ChanceNode:
    """Child: wet_grass | rain, indexed by rain's integer value."""
    return ChanceNode(
        name="wet_grass",
        parents=("rain",),
        states=("dry", "wet"),
        dist=lambda rain: dist.Categorical(probs=_P_WET[rain]),
    )


def _rain_wet_grass_diagram() -> InfluenceDiagram:
    """A valid two-node BN: rain -> wet_grass, both configured."""
    diag = InfluenceDiagram()
    diag.add_node(_rain_node())
    diag.add_node(_wet_grass_node())
    return diag


def _empty_diagram() -> InfluenceDiagram:
    """A diagram with no nodes — still snapshots to a valid (trivial) model."""
    return InfluenceDiagram()


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


# --- to_model ---------------------------------------------------------------


def test_model_is_callable() -> None:
    """to_model returns a callable taking no args."""
    model = to_model(_rain_wet_grass_diagram().snapshot())
    assert callable(model)


def test_model_returns_dict() -> None:
    """The model function returns a dict (not None) keyed by node name."""
    model = to_model(_rain_wet_grass_diagram().snapshot())
    seeded = numpyro.handlers.seed(model, jax_random_key())
    result = seeded()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_translates_empty_snapshot() -> None:
    """A zero-node snapshot yields a model that traces to no sample sites."""
    model = to_model(_empty_diagram().snapshot())
    trace = _trace(model)
    assert trace.keys() == set()


def test_topological_order_respected() -> None:
    """Every sample site named after a node; parents resolve before children."""
    model = to_model(_rain_wet_grass_diagram().snapshot())
    trace = _trace(model)
    assert set(trace.keys()) == {"rain", "wet_grass"}


# --- forward sampling (the end-to-end proof) --------------------------------


def test_forward_sample_returns_named_sites() -> None:
    """Predictive produces one array per node, keyed by node name."""
    model = to_model(_rain_wet_grass_diagram().snapshot())
    samples = Predictive(model, num_samples=1000)(jax_random_key())
    assert set(samples.keys()) == {"rain", "wet_grass"}
    assert samples["rain"].shape == (1000,)
    assert samples["wet_grass"].shape == (1000,)


def test_forward_sample_marginal_matches_handcomputed() -> None:
    """
    Empirical marginals match the hand-computed values.

    rain=yes has prior 0.2. P(wet_grass=wet) marginalizes over rain:

        P(wet) = P(wet|no)*P(no) + P(wet|yes)*P(yes)
               = 0.20*0.8 + 0.95*0.2 = 0.35

    Forward sampling must reproduce both within sampling tolerance.
    """
    model = to_model(_rain_wet_grass_diagram().snapshot())
    samples = Predictive(model, num_samples=20_000)(jax_random_key())

    # rain: index 1 == 'yes'. Prior P(yes) = 0.2.
    p_rain_yes = float(jnp.mean(samples["rain"] == 1))
    # wet_grass: index 1 == 'wet'. Hand-computed P(wet) = 0.35.
    p_wet = float(jnp.mean(samples["wet_grass"] == 1))

    assert p_rain_yes == pytest.approx(0.2, abs=0.02)
    assert p_wet == pytest.approx(0.35, abs=0.02)


def test_forward_sample_continuous_root() -> None:
    """A continuous root node samples to plausible support, not indices."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="height",
            dist=lambda: dist.Normal(loc=170.0, scale=5.0),
        )
    )
    model = to_model(diag.snapshot())
    samples = Predictive(model, num_samples=20_000)(jax_random_key())
    heights = samples["height"]
    assert heights.shape == (20_000,)
    assert float(jnp.mean(heights)) == pytest.approx(170.0, abs=0.5)
    assert float(jnp.std(heights)) == pytest.approx(5.0, abs=0.5)


# --- posterior inference helpers -------------------------------------------


def _posterior_samples(
    model: Any,
    *,
    num_samples: int = 2000,
    base_seed: int = 0,
) -> dict[str, Any]:
    """
    Collect posterior samples for a model with enumerated discrete latents.

    For each sample, calls ``infer_discrete`` (which computes the exact
    posterior) and reads the substituted value from the trace.  Returns a
    dict mapping site name to a 1-D JAX array of ``num_samples`` values.
    """
    import jax

    values: dict[str, list[Any]] = {}
    for i in range(num_samples):
        rng = jax.random.PRNGKey(base_seed + i)
        inferred = infer_discrete(model, temperature=1, rng_key=rng)
        tr = numpyro.handlers.trace(inferred).get_trace()
        for name, site in tr.items():
            if site["type"] == "sample":
                values.setdefault(name, []).append(site["value"])
    return {name: jnp.array(vals) for name, vals in values.items()}


# --- posterior inference: discrete ------------------------------------------


def test_posterior_discrete_with_observation() -> None:
    """
    infer_discrete recovers exact posterior on rain given wet_grass=wet.

    Hand-computed:
        P(rain=yes | wet) = P(wet|yes)*P(yes) / P(wet)
                          = 0.95 * 0.2 / 0.35 ≈ 0.543
    """
    model = to_model(
        _rain_wet_grass_diagram().snapshot(),
        observed={"wet_grass": jnp.array(1)},
    )
    samples = _posterior_samples(model, num_samples=2000)
    p_rain_yes = float(jnp.mean(samples["rain"] == 1))
    assert p_rain_yes == pytest.approx(0.543, abs=0.03)


def test_posterior_discrete_all_latent_with_default_observed() -> None:
    """Forward sampling still works when observed=None (the default)."""
    model = to_model(_rain_wet_grass_diagram().snapshot())
    samples = Predictive(model, num_samples=500)(jax_random_key())
    assert set(samples.keys()) == {"rain", "wet_grass"}


# --- posterior inference: continuous ----------------------------------------


def test_posterior_continuous_hierarchical() -> None:
    """
    MCMC recovers mu posterior given height observation.

    Model: mu ~ Normal(0, 10); height ~ Normal(mu, 5). Observe height=175.
    Posterior: mu ~ Normal(175 * 100 / (100+25), sqrt(1/(1/100 + 1/25)))
             = Normal(140, ~4.47)
    """
    model = to_model(
        _continuous_hierarchical_diagram().snapshot(),
        observed={"height": jnp.array(175.0)},
    )
    mcmc = MCMC(
        NUTS(model),
        num_warmup=500,
        num_samples=2000,
        progress_bar=False,
    )
    mcmc.run(jax_random_key())
    posterior = mcmc.get_samples()

    mu_samples = posterior["mu"]
    assert mu_samples.shape == (2000,)
    # Normal-Normal conjugate: posterior mean = 140, posterior std ≈ 4.47
    assert float(jnp.mean(mu_samples)) == pytest.approx(140.0, abs=1.0)
    assert float(jnp.std(mu_samples)) == pytest.approx(4.47, abs=0.5)


def test_posterior_mixed_discrete_continuous() -> None:
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

    The discrete node is auto-detected and enumerated. Predictive with
    parallel=True computes the exact posterior over the enumerated latent.
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
    model = to_model(diag.snapshot(), observed={"height": jnp.array(175.0)})
    samples = _posterior_samples(model, num_samples=2000)
    p_rain_yes = float(jnp.mean(samples["rain"] == 1))
    assert p_rain_yes == pytest.approx(0.292, abs=0.03)


# --- errors -----------------------------------------------------------------


def test_missing_numpyro_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If numpyro cannot be imported, to_model raises an actionable ImportError."""
    import sys

    monkeypatch.setitem(sys.modules, "numpyro", None)
    snap = _rain_wet_grass_diagram().snapshot()
    with pytest.raises(ImportError, match="numpyro"):
        to_model(snap)


# --- small driver helpers ---------------------------------------------------


def _trace(model: Any) -> dict[str, Any]:
    """Run a model under a seeded trace and return its named sample sites."""
    import numpyro

    seeded = numpyro.handlers.seed(model, jax_random_key())
    return numpyro.handlers.trace(seeded).get_trace()


def jax_random_key(seed: int = 0) -> Any:
    """A fixed PRNG key for reproducible sampling in tests."""
    import jax

    return jax.random.PRNGKey(seed)
