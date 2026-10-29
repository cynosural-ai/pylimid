"""Distribution metadata, constraint transforms, and continuous fitted rules."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints, transforms

from pylimid import (
    ContinuousPolicy,
    DistributionPolicy,
    Interval,
    Linear,
    PolicyFamily,
    Real,
)
from pylimid._numpyro.policies import policy_distribution


def _rule(family, domain=Real()):
    parameters = family.initialize(
        jax.random.PRNGKey(0), num_actions=None, observations={}, parent_states={}
    )
    return ContinuousPolicy(family, (), (), (), domain, parameters)


@pytest.mark.parametrize(
    ("constructor", "names", "values"),
    [
        (dist.Normal, ("loc", "scale"), (0.3, 0.7)),
        (dist.LogNormal, ("loc", "scale"), (0.3, 0.7)),
        (dist.Beta, ("concentration1", "concentration0"), (2.0, 3.0)),
        (dist.Gamma, ("concentration", "rate"), (2.0, 3.0)),
    ],
)
def test_automatic_outputs_follow_numpyro_parameter_names_and_constraints(
    constructor, names, values
):
    family = DistributionPolicy(constructor, Linear())
    rule = _rule(family)
    assert rule.parameters["bias"].shape == (2,)
    raw = jnp.array(
        [
            value if name == "loc" else jnp.log(value)
            for name, value in zip(names, values, strict=True)
        ]
    )
    rule = replace(rule, parameters={"weights": jnp.empty((0, 2)), "bias": raw})
    distribution = rule.distribution()
    for name, value in zip(names, values, strict=True):
        assert getattr(distribution, name) == pytest.approx(value)
    draws = jax.vmap(rule.sample)(jax.random.split(jax.random.PRNGKey(1), 32))
    assert draws.shape == (32,)
    assert jnp.all(jnp.isfinite(draws))


def test_explicit_output_order_and_fixed_parameters_are_preserved():
    family = DistributionPolicy(dist.Normal, Linear(outputs=("scale", "loc")))
    rule = _rule(family)
    rule = replace(
        rule,
        parameters={
            "weights": jnp.empty((0, 2)),
            "bias": jnp.array([jnp.log(0.4), 1.2]),
        },
    )
    assert rule.distribution().loc == pytest.approx(1.2)
    assert rule.distribution().scale == pytest.approx(0.4)

    fixed = {"scale": 0.3}
    family = DistributionPolicy(
        dist.Normal, Linear(outputs=("loc",)), fixed_parameters=fixed
    )
    fixed["scale"] = -1.0
    rule = _rule(family)
    assert rule.parameters["bias"].shape == (1,)
    assert rule.distribution().scale == pytest.approx(0.3)


def test_linear_model_encodes_mixed_information_in_declared_order():
    family = DistributionPolicy(dist.Normal, Linear(), fixed_parameters={"scale": 0.5})
    rule = ContinuousPolicy(
        family,
        ("category", "signal"),
        (("off", "on"), None),
        ((), (1, 2)),
        Real(),
        {"weights": jnp.array([[1.0], [2.0], [3.0], [4.0]]), "bias": jnp.array([0.5])},
    )
    assert rule.distribution(
        signal=jnp.array([[2.0, 3.0]]), category=jnp.array(1)
    ).mean == pytest.approx(20.5)
    with pytest.raises(TypeError, match="Unexpected information"):
        rule.sample(
            jax.random.PRNGKey(0),
            category=jnp.array(0),
            signal=jnp.zeros((1, 2)),
            hidden=jnp.array(0),
        )
    with pytest.raises(KeyError, match="category"):
        rule.distribution(signal=jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="must have shape"):
        rule.distribution(category=jnp.array(0), signal=jnp.zeros(2))


@pytest.mark.parametrize(
    ("outputs", "fixed"),
    [
        (("loc",), {}),
        (("unknown", "scale"), {}),
        (("loc", "scale"), {"scale": 1.0}),
        (None, {"unknown": 1.0}),
    ],
)
def test_parameter_selection_must_cover_the_distribution_without_overlap(
    outputs, fixed
):
    with pytest.raises(
        ValueError, match="cover exactly|both learned and fixed|Unknown fixed"
    ):
        DistributionPolicy(dist.Normal, Linear(outputs=outputs), fixed_parameters=fixed)


@pytest.mark.parametrize("scale", [-1.0, 0.0, float("nan"), float("inf")])
def test_fixed_parameters_must_satisfy_their_constraints(scale):
    with pytest.raises(ValueError, match="constraint|finite"):
        DistributionPolicy(dist.Normal, Linear(), fixed_parameters={"scale": scale})


@pytest.mark.parametrize("scale", [True, 1j, [1.0]])
def test_fixed_parameters_must_be_real_scalars(scale):
    with pytest.raises(TypeError, match="real scalar"):
        DistributionPolicy(dist.Normal, Linear(), fixed_parameters={"scale": scale})


@pytest.mark.parametrize(
    "constructor", [dist.Categorical, dist.BernoulliProbs, dist.Dirichlet, dist.Uniform]
)
def test_automatic_distribution_adapter_rejects_unsupported_metadata(constructor):
    with pytest.raises(NotImplementedError, match="metadata|scalar continuous support"):
        DistributionPolicy(constructor, Linear())


def test_bound_normal_requires_explicit_fixed_truncation():
    with pytest.raises(ValueError, match="requires fixed"):
        DistributionPolicy(dist.TruncatedNormal, Linear())
    with pytest.raises(ValueError, match="stay within"):
        _rule(
            DistributionPolicy(dist.Normal, Linear()), Interval(0.0, 1.0)
        ).distribution()
    family = DistributionPolicy(
        dist.TruncatedNormal,
        Linear(),
        fixed_parameters={"low": 0.0, "high": 1.0, "scale": 0.3},
    )
    rule = _rule(family, Interval(-1.0, 2.0))
    draws = jax.vmap(rule.sample)(jax.random.split(jax.random.PRNGKey(3), 128))
    assert jnp.all((draws >= 0.0) & (draws <= 1.0))
    assert jnp.isneginf(rule.log_prob(jnp.array(1.5)))


class _ScaledBeta:
    """NumPyro-compatible factory with explicit scalar support metadata."""

    arg_constraints = dist.Beta.arg_constraints
    support = constraints.interval(2.0, 4.0)

    def __call__(self, concentration1, concentration0):
        return dist.TransformedDistribution(
            dist.Beta(concentration1, concentration0),
            transforms.AffineTransform(2.0, 2.0, domain=constraints.unit_interval),
        )


def test_custom_bounded_factory_composes_with_the_same_parameter_model():
    rule = _rule(DistributionPolicy(_ScaledBeta(), Linear()), Interval(2.0, 4.0))
    # Zero raw concentrations give Beta(1, 1), hence a uniform density of 1/2.
    assert rule.log_prob(jnp.array(3.0)) == pytest.approx(-jnp.log(2.0), abs=2e-6)
    draws = jax.vmap(rule.sample)(jax.random.split(jax.random.PRNGKey(4), 128))
    assert jnp.all((draws >= 2.0) & (draws <= 4.0))


def test_transformed_distribution_must_report_its_actual_bounded_domain():
    class MissingTransformDomain(_ScaledBeta):
        def __call__(self, concentration1, concentration0):
            return dist.TransformedDistribution(
                dist.Beta(concentration1, concentration0),
                transforms.AffineTransform(2.0, 2.0),
            )

    with pytest.raises(ValueError, match="stay within"):
        _rule(
            DistributionPolicy(MissingTransformDomain(), Linear()), Interval(2.0, 4.0)
        ).distribution()


class _DistributionFamily(PolicyFamily):
    """Return a supplied distribution for execution-contract checks."""

    def __init__(self, distribution):
        self.action_distribution = distribution

    def initialize(self, rng_key, *, num_actions, observations, parent_states):
        return {}

    def distribution(self, parameters, observations, parent_states):
        return self.action_distribution


@pytest.mark.parametrize(
    "distribution",
    [
        dist.Normal(jnp.zeros(2), 1.0),
        dist.Normal(jnp.zeros(2), 1.0).to_event(1),
        dist.Bernoulli(0.5),
        dist.Delta(0.0),
    ],
)
def test_continuous_rules_require_scalar_continuous_densities(distribution):
    with pytest.raises(
        ValueError, match="scalar distribution|continuous action density"
    ):
        _rule(_DistributionFamily(distribution)).distribution()


def test_parameter_dependent_support_is_rejected_during_differentiation():
    class MovingSupport(_DistributionFamily):
        def distribution(self, parameters, observations, parent_states):
            return dist.Uniform(parameters["low"], parameters["low"] + 1.0)

    rule = ContinuousPolicy(
        MovingSupport(None), (), (), (), Real(), {"low": jnp.array(0.0)}
    )
    with pytest.raises(ValueError, match="support bounds must be fixed"):
        jax.grad(
            lambda low: replace(rule, parameters={"low": low}).log_prob(jnp.array(0.5))
        )(jnp.array(0.0))


def test_parameter_model_must_return_one_scalar_per_output():
    class WrongShape(Linear):
        def predict(self, parameters, observations, parent_states):
            return jnp.zeros((2, 1))

    with pytest.raises(ValueError, match="Parameter model must return shape"):
        _rule(DistributionPolicy(dist.Normal, WrongShape())).distribution()


@pytest.mark.parametrize(
    "constructor",
    [
        dist.Normal,
        dist.Beta,
        dist.Gamma,
        dist.LogNormal,
        dist.TruncatedNormal,
        _ScaledBeta(),
    ],
)
def test_broadcast_policy_distributions_have_independent_batched_draws(constructor):
    fixed = {"low": 0.0, "high": 1.0} if constructor is dist.TruncatedNormal else {}
    family = DistributionPolicy(constructor, Linear(), fixed_parameters=fixed)
    rule = ContinuousPolicy(
        family,
        ("category", "signal"),
        (("off", "on"), None),
        ((), ()),
        Real(),
        {
            "weights": jnp.array([[0.1, 0.2], [-0.1, 0.1], [0.2, -0.1]]),
            "bias": jnp.zeros(2),
        },
    )
    distribution = policy_distribution(
        rule,
        {"category": jnp.array([0, 1]), "signal": jnp.array([[-0.5], [0.0], [0.5]])},
    )
    assert distribution.batch_shape == (3, 2)
    draws = jnp.asarray(distribution.sample(jax.random.PRNGKey(19), (2000,)))
    assert draws.shape == (2000, 3, 2)
    assert jnp.shape(distribution.log_prob(draws)) == (2000, 3, 2)
    correlation = jnp.corrcoef(draws.reshape(2000, 6).T)
    assert jnp.max(jnp.abs(correlation - jnp.eye(6))) < 0.15
