"""Parameter-model output names and custom nested parameter trees."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid.graph import Real
from pylimid.policy import ContinuousPolicy, DistributionPolicy, Linear, ParameterModel


@pytest.mark.parametrize("outputs", [("loc", "loc"), ("",), (1,), ["loc", "scale"]])
def test_parameter_model_validates_output_names(outputs):
    with pytest.raises((TypeError, ValueError), match="outputs|Output names"):
        Linear(outputs=outputs)


def test_custom_models_can_initialize_nested_parameter_trees():
    class Quadratic(ParameterModel):
        def initialize(self, rng_key, *, num_outputs, observations, parent_states):
            return {"layer": {"weights": jnp.zeros(num_outputs)}}

        def predict(self, parameters, observations, parent_states):
            return parameters["layer"]["weights"] * observations["signal"] ** 2

    family = DistributionPolicy(
        dist.Normal, Quadratic(), fixed_parameters={"scale": 0.5}
    )
    rule = ContinuousPolicy(
        family,
        ("signal",),
        (None,),
        ((),),
        Real(),
        {"layer": {"weights": jnp.array([2.0])}},
    )
    assert rule.distribution(signal=jnp.array(3.0)).mean == pytest.approx(18.0)
