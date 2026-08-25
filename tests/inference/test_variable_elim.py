"""Tests for :mod:`decisionpy.inference.exact.categorical` — variable elimination."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.inference.exact.categorical import query

# --- helpers ----------------------------------------------------------------


def _prior_rain() -> list[float]:
    return [0.8, 0.2]


def _cpt_wet() -> list[list[float]]:
    return [
        [0.80, 0.20],  # rain=no  -> P(dry)=0.8, P(wet)=0.2
        [0.05, 0.95],  # rain=yes -> P(dry)=0.05, P(wet)=0.95
    ]


def _rain_wet_grass_diagram() -> InfluenceDiagram:
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="rain",
            states=("no", "yes"),
            dist=lambda: dist.Categorical(probs=jnp.array(_prior_rain())),
        )
    )
    diag.add_node(
        ChanceNode(
            name="wet_grass",
            parents=("rain",),
            states=("dry", "wet"),
            dist=lambda rain: dist.Categorical(probs=jnp.array(_cpt_wet()[rain])),
        )
    )
    return diag


# --- prior marginals ---------------------------------------------------------


def test_prior_single_root() -> None:
    """Prior P(rain) = [0.8, 0.2]."""
    result = query(_rain_wet_grass_diagram().snapshot(), variables=["rain"])
    assert result["rain"] == pytest.approx([0.8, 0.2])


def test_prior_child() -> None:
    """Prior P(wet_grass) = [0.65, 0.35] (hand-computed)."""
    # P(dry) = P(dry|no)*P(no) + P(dry|yes)*P(yes) = 0.80*0.8 + 0.05*0.2 = 0.65
    # P(wet) = 1 - 0.65 = 0.35
    result = query(_rain_wet_grass_diagram().snapshot(), variables=["wet_grass"])
    assert result["wet_grass"] == pytest.approx([0.65, 0.35])


def test_prior_joint() -> None:
    """Querying both variables returns both marginals in one call."""
    result = query(
        _rain_wet_grass_diagram().snapshot(), variables=["rain", "wet_grass"]
    )
    assert result["rain"] == pytest.approx([0.8, 0.2])
    assert result["wet_grass"] == pytest.approx([0.65, 0.35])


def test_prior_empty_diagram() -> None:
    """A diagram with no nodes returns a unit factor."""
    diag = InfluenceDiagram()
    result = query(diag.snapshot(), variables=[])
    assert result == {}


# --- posterior (with evidence) -----------------------------------------------


def test_posterior_rain_given_wet() -> None:
    """
    P(rain | wet_grass=wet) — hand-computed.

    P(yes|wet) = P(wet|yes)*P(yes) / P(wet) = 0.95*0.2 / 0.35 ≈ 0.5429
    P(no|wet)  = 1 - 0.5429 ≈ 0.4571
    """
    result = query(
        _rain_wet_grass_diagram().snapshot(),
        variables=["rain"],
        observed={"wet_grass": 1},
    )
    assert result["rain"] == pytest.approx([0.457142857, 0.542857143])


def test_posterior_child_given_parent() -> None:
    """P(wet_grass | rain=yes) should just be the CPT row [0.05, 0.95]."""
    result = query(
        _rain_wet_grass_diagram().snapshot(),
        variables=["wet_grass"],
        observed={"rain": 0},
    )
    assert result["wet_grass"] == pytest.approx([0.80, 0.20])


def test_posterior_irrelevant_evidence() -> None:
    """Evidence on an independent branch doesn't affect the other branch."""
    # a -> b, c (independent). P(a | c=0) should equal P(a).
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="a",
            states=("x", "y"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.3, 0.7])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="b",
            parents=("a",),
            states=("u", "v"),
            dist=lambda a: dist.Categorical(
                probs=jnp.array([[0.9, 0.1], [0.4, 0.6]])[a]
            ),
        )
    )
    diag.add_node(
        ChanceNode(
            name="c",
            states=("p", "q"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
        )
    )
    result = query(
        diag.snapshot(),
        variables=["a"],
        observed={"c": 0},
    )
    assert result["a"] == pytest.approx([0.3, 0.7])


# --- three-node chains -------------------------------------------------------


def test_chain_three_nodes() -> None:
    """A → b → c. Hand-check P(c) and P(a | c=c1)."""
    # a ~ [0.4, 0.6]
    # b|a: [[0.7, 0.3], [0.2, 0.8]]
    # c|b: [[0.9, 0.1], [0.5, 0.5]]
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="a",
            states=("a0", "a1"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.4, 0.6])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="b",
            parents=("a",),
            states=("b0", "b1"),
            dist=lambda a: dist.Categorical(
                probs=jnp.array([[0.7, 0.3], [0.2, 0.8]])[a]
            ),
        )
    )
    diag.add_node(
        ChanceNode(
            name="c",
            parents=("b",),
            states=("c0", "c1"),
            dist=lambda b: dist.Categorical(
                probs=jnp.array([[0.9, 0.1], [0.5, 0.5]])[b]
            ),
        )
    )

    # P(c=c0) manual:
    # P(c0) = P(c0|b0)*P(b0) + P(c0|b1)*P(b1)
    # P(b0) = P(b0|a0)*P(a0) + P(b0|a1)*P(a1) = 0.7*0.4 + 0.2*0.6 = 0.28+0.12=0.40
    # P(b1) = 0.60
    # P(c0) = 0.9*0.4 + 0.5*0.6 = 0.36 + 0.30 = 0.66
    # P(c1) = 0.34
    result = query(diag.snapshot(), variables=["c"])
    assert result["c"] == pytest.approx([0.66, 0.34])

    # Posterior P(a | c=c1):
    # P(a0|c1) = P(c1|a0) * P(a0) / P(c1)
    # P(c1|a0) = P(c1|b0)*P(b0|a0) + P(c1|b1)*P(b1|a0) = 0.1*0.7+0.5*0.3=0.22
    # P(c1|a1) = 0.1*0.2+0.5*0.8=0.42
    # P(c1) = 0.22*0.4 + 0.42*0.6 = 0.088+0.252=0.34
    # P(a0|c1) = 0.22*0.4 / 0.34 ≈ 0.2588
    # P(a1|c1) = 1 - 0.2588 ≈ 0.7412
    post = query(diag.snapshot(), variables=["a"], observed={"c": 1})
    assert post["a"] == pytest.approx([0.2588235, 0.7411765])


# --- multi-parent ------------------------------------------------------------


def test_converging_parents() -> None:
    """A → c ← b (v-structure). Check explaining-away P(a | b=b1, c=c0)."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="a",
            states=("a0", "a1"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    diag.add_node(
        ChanceNode(
            name="b",
            states=("b0", "b1"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
        )
    )
    # c depends on both a and b.
    # Rows: a0, a1; Cols: b0, b1
    cpt_c = jnp.array(
        [
            [[0.95, 0.05], [0.30, 0.70]],  # a=a0: [b=b0, b=b1]
            [[0.20, 0.80], [0.01, 0.99]],  # a=a1
        ]
    )
    diag.add_node(
        ChanceNode(
            name="c",
            parents=("a", "b"),
            states=("c0", "c1"),
            dist=lambda a, b: dist.Categorical(probs=cpt_c[a][b]),
        )
    )

    # P(c=c0) by hand:
    # P(c0) = Σ   P(c0|a,b) * P(a) * P(b)
    # = 0.95*0.5*0.6 + 0.30*0.5*0.4 + 0.20*0.5*0.6 + 0.01*0.5*0.4
    # = 0.285 + 0.060 + 0.060 + 0.002 = 0.407
    result = query(diag.snapshot(), variables=["c"])
    assert result["c"] == pytest.approx([0.407, 0.593])

    # Posterior P(a | b=b1, c=c0):
    # P(a0 | b1,c0) = P(c0|a0,b1)*P(b1)*P(a0) / P(c0,b1)
    # Need P(c0,b1) = Σ P(c0|a,b1)*P(a)*P(b1) = 0.30*0.5*0.4 + 0.01*0.5*0.4 = 0.062
    # P(a0|b1,c0) = 0.30*0.4*0.5 / 0.062 = 0.06/0.062 ≈ 0.9677
    post = query(
        diag.snapshot(),
        variables=["a"],
        observed={"b": 1, "c": 0},
    )
    pa0 = 0.30 * 0.5 / (0.30 * 0.5 + 0.01 * 0.5)  # marginalizes b since it's fixed
    assert post["a"] == pytest.approx([pa0, 1 - pa0])


# --- errors ------------------------------------------------------------------


def test_continuous_node_raises() -> None:
    """VE requires all-discrete nodes; a continuous node raises TypeError."""
    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0, 1)))
    with pytest.raises(TypeError, match="all-discrete"):
        query(diag.snapshot(), variables=["x"])


# --- vs exact values (double-check with math) --------------------------------


def test_single_node_observed() -> None:
    """Observing a root node makes it deterministic (no inference needed)."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="coin",
            states=("h", "t"),
            dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])),
        )
    )
    result = query(diag.snapshot(), variables=["coin"], observed={"coin": 0})
    assert result["coin"] == pytest.approx([1.0, 0.0])


def test_prior_sums_to_one() -> None:
    """Every marginal should sum to 1.0."""
    result = query(
        _rain_wet_grass_diagram().snapshot(), variables=["rain", "wet_grass"]
    )
    assert math.isclose(sum(result["rain"]), 1.0)
    assert math.isclose(sum(result["wet_grass"]), 1.0)


def test_posterior_sums_to_one() -> None:
    result = query(
        _rain_wet_grass_diagram().snapshot(),
        variables=["rain"],
        observed={"wet_grass": 1},
    )
    assert math.isclose(sum(result["rain"]), 1.0)


def test_query_all_variables() -> None:
    """Querying all variables returns correct marginals for each."""
    result = query(
        _rain_wet_grass_diagram().snapshot(), variables=["rain", "wet_grass"]
    )
    assert result["rain"] == pytest.approx([0.8, 0.2])
    assert result["wet_grass"] == pytest.approx([0.65, 0.35])


def test_observed_query_same_variable() -> None:
    """Querying a variable that is also observed returns a point mass."""
    result = query(
        _rain_wet_grass_diagram().snapshot(),
        variables=["wet_grass"],
        observed={"wet_grass": 1, "rain": 0},
    )
    assert result["wet_grass"] == pytest.approx([0.0, 1.0])


def test_dist_without_probs_raises() -> None:
    """The contract is ``probs``: a dist object without one fails loudly."""

    class BareLogProb:
        def log_prob(self, value: int) -> float:
            return 0.0

    diag = InfluenceDiagram()
    diag.add_node(ChanceNode(name="x", states=("a", "b"), dist=lambda: BareLogProb()))
    with pytest.raises(RuntimeError, match="probs"):
        query(diag.snapshot(), variables=["x"])
