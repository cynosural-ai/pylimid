"""NumPyro model translation, sample sites, and enumeration annotations."""

from __future__ import annotations

from typing import Any

import jax
import numpyro
import numpyro.distributions as dist
import pytest

from pylimid._numpyro.model import build_model
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode
from tests._numpyro.diagrams import rain_wet_grass_diagram


def test_model_returns_dict() -> None:
    """The model function returns a dict (not None) keyed by node name."""
    model = build_model(rain_wet_grass_diagram().snapshot(), {}, {})
    seeded = numpyro.handlers.seed(model, _jax_random_key())
    result = seeded()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"rain", "wet_grass"}


def test_translates_empty_snapshot() -> None:
    """A zero-node snapshot yields a model that traces to no sample sites."""
    model = build_model(_empty_diagram().snapshot(), {}, {})
    trace = _trace(model)
    assert trace.keys() == set()


def test_topological_order_respected() -> None:
    """A child inserted first executes after its parent in the model trace."""
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="child",
            parents=("parent",),
            dist=lambda parent: dist.Normal(parent, 1.0),
        )
    )
    diagram.add_node(ChanceNode(name="parent", dist=lambda: dist.Normal(0.0, 1.0)))

    model = build_model(diagram.snapshot(), {}, {})
    trace = _trace(model)

    assert list(trace) == ["parent", "child"]


@pytest.mark.parametrize(
    "distribution,enumerable",
    [
        (lambda: dist.Binomial(total_count=10, probs=0.2), True),
        (lambda: dist.Poisson(3.0), False),
    ],
)
def test_count_enumeration_annotation_follows_distribution(distribution, enumerable):
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="count", dist=distribution))
    model = build_model(diagram.snapshot(), {}, {})

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace()

    if enumerable:
        assert trace["count"]["infer"]["enumerate"] == "parallel"
    else:
        assert "enumerate" not in trace["count"]["infer"]


def _empty_diagram() -> InfluenceDiagram:
    """A diagram with no nodes — still snapshots to a valid (trivial) model."""
    return InfluenceDiagram()


def _trace(model: Any) -> dict[str, Any]:
    """Run a model under a seeded trace and return its named sample sites."""
    seeded = numpyro.handlers.seed(model, _jax_random_key())
    return numpyro.handlers.trace(seeded).get_trace()


def _jax_random_key(seed: int = 0) -> Any:
    """A fixed PRNG key for reproducible sampling in tests."""
    return jax.random.PRNGKey(seed)
