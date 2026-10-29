"""Shared discrete diagrams for NumPyro backend tests."""

import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode

# rain ~ Bernoulli(0.2); wet_grass | rain uses one conditional row per state.
_PRIOR_RAIN = jnp.array([0.8, 0.2])
_P_WET = jnp.array([[0.80, 0.20], [0.05, 0.95]])


def rain_node() -> ChanceNode:
    """Root node: rain ~ Categorical([0.8, 0.2])."""
    return ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=_PRIOR_RAIN),
    )


def rain_wet_grass_diagram() -> InfluenceDiagram:
    """A valid two-node BN: rain -> wet_grass, both configured."""
    diag = InfluenceDiagram()
    diag.add_node(rain_node())
    diag.add_node(_wet_grass_node())
    return diag


def _wet_grass_node() -> ChanceNode:
    """Child: wet_grass | rain, indexed by rain's integer value."""
    return ChanceNode(
        name="wet_grass",
        parents=("rain",),
        states=("dry", "wet"),
        dist=lambda rain: dist.Categorical(probs=_P_WET[rain]),
    )
