"""
``is_solvable`` implements the exact-solution-ordering criterion.

Two layers: the shared LIMID fixtures (full models) and a battery of
structural cases covering the subtleties of the criterion — decisions that
do not interact despite sharing info (separate utilities), influences
d-separated by an observed variable, forgotten irrelevant observations, and
genuinely unordered decisions.

The battery agrees with pyAgrum's ``isSolvable()`` on every case except the
documented divergence: pyAgrum only compares decisions within a level of its
partial order, so it can admit a higher-level decision influencing a
lower-level decision's utilities unobserved. The divergence is pinned below.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpyro.distributions as dist
import pyagrum as gum
import pytest

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from pylimid.inference.numpyro.solvers.regularity import (
    is_solvable,
    solvability_order,
)

from .._limid_fixtures import (
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    SingleDecisionTreat,
    ThreeStateClimate,
    TwoDecisionsIndependent,
)

FIXTURES = {
    "SingleDecisionTreat": SingleDecisionTreat,
    "TwoDecisionsIndependent": TwoDecisionsIndependent,
    "EmptyInfoDecision": EmptyInfoDecision,
    "IrrelevantInfoDecision": IrrelevantInfoDecision,
    "NestedTwoDecisions": NestedTwoDecisions,
    "ThreeStateClimate": ThreeStateClimate,
}


def _pyagrum_is_solvable(diagram) -> bool:
    return gum.ShaferShenoyLIMIDInference(diagram).isSolvable()  # ty: ignore[possibly-missing-attribute]


@pytest.mark.parametrize("fixture", FIXTURES.values(), ids=list(FIXTURES))
def test_matches_pyagrum_on_fixtures(fixture):
    snapshot = fixture.pylimid().snapshot()
    assert is_solvable(snapshot) is _pyagrum_is_solvable(fixture.pyagrum())
    assert (solvability_order(snapshot) is not None) is is_solvable(snapshot)


# ---------------------------------------------------------------------------
# Structural battery
# ---------------------------------------------------------------------------


def _pylimid(edges, states, kinds) -> InfluenceDiagram:
    diag = InfluenceDiagram()
    for name, node_states in states.items():
        parents = tuple(parent for parent, child in edges if child == name)
        if kinds[name] == "chance":
            probs = jnp.ones(len(node_states)) / len(node_states)
            diag.add_node(
                ChanceNode(
                    name=name,
                    parents=parents,
                    states=node_states,
                    dist=_uniform_factory(parents, probs),
                )
            )
        elif kinds[name] == "decision":
            diag.add_node(DecisionNode(name=name, parents=parents, states=node_states))
        else:
            diag.add_node(
                UtilityNode(
                    name=name, parents=parents, values=_constant_factory(parents)
                )
            )
    return diag


def _uniform_factory(parents, probs):
    def fn(**kwargs):
        return dist.Categorical(probs=probs)

    return _declare_signature(fn, parents)


def _constant_factory(parents):
    def fn(**kwargs):
        return 0.0

    return _declare_signature(fn, parents)


def _declare_signature(fn, names):
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    return fn


def _pyagrum(edges, states, kinds) -> gum.InfluenceDiagram:  # ty: ignore[possibly-missing-attribute]
    g = gum.InfluenceDiagram()  # ty: ignore[possibly-missing-attribute]
    for name, node_states in states.items():
        var = gum.LabelizedVariable(name, name, list(node_states))
        add = {
            "chance": g.addChanceNode,
            "decision": g.addDecisionNode,
            "utility": g.addUtilityNode,
        }[kinds[name]]
        add(var)
    for parent, child in edges:
        g.addArc(g.idFromName(parent), g.idFromName(child))
    return g


def _case(edges, states, kinds, expected):
    return edges, states, kinds, expected


BATTERY = {
    # The decisions share an information parent but have separate utilities:
    # they do not interact, so no ordering is needed.
    "separate_utilities_shared_info": _case(
        [("X", "D1"), ("X", "D2"), ("D1", "U1"), ("D2", "U2")],
        {
            "X": ("a", "b"),
            "D1": ("a", "b"),
            "D2": ("a", "b"),
            "U1": ("p",),
            "U2": ("p",),
        },
        {
            "X": "chance",
            "D1": "decision",
            "D2": "decision",
            "U1": "utility",
            "U2": "utility",
        },
        True,
    ),
    # D1 influences D2's utility through Y, but Y is observed by D2 — the
    # influence is d-separated by the information set.
    "influence_chain_separate_utilities": _case(
        [("D1", "Y"), ("Y", "D2"), ("D1", "U1"), ("Y", "U2"), ("D2", "U2")],
        {
            "D1": ("a", "b"),
            "Y": ("a", "b"),
            "D2": ("a", "b"),
            "U1": ("p",),
            "U2": ("p",),
        },
        {
            "D1": "decision",
            "Y": "chance",
            "D2": "decision",
            "U1": "utility",
            "U2": "utility",
        },
        True,
    ),
    # D2 forgot X, but X influences no utility: still solvable.
    "forgotten_irrelevant_info": _case(
        [("X", "D1"), ("D1", "D2"), ("D1", "U"), ("D2", "U")],
        {"X": ("a", "b"), "D1": ("a", "b"), "D2": ("a", "b"), "U": ("p",)},
        {"X": "chance", "D1": "decision", "D2": "decision", "U": "utility"},
        True,
    ),
    # A shared utility without an ordering: the decisions cannot be
    # coordinated.
    "shared_utility_no_ordering": _case(
        [("X", "D1"), ("X", "D2"), ("D1", "U"), ("D2", "U")],
        {"X": ("a", "b"), "D1": ("a", "b"), "D2": ("a", "b"), "U": ("p",)},
        {"X": "chance", "D1": "decision", "D2": "decision", "U": "utility"},
        False,
    ),
    "shared_utility_root_decisions": _case(
        [("D1", "U"), ("D2", "U")],
        {"D1": ("a", "b"), "D2": ("a", "b"), "U": ("p",)},
        {"D1": "decision", "D2": "decision", "U": "utility"},
        False,
    ),
    # A no-forgetting chain of three decisions.
    "three_decision_chain": _case(
        [
            ("X", "D1"),
            ("X", "D2"),
            ("D1", "D2"),
            ("X", "D3"),
            ("D1", "D3"),
            ("D2", "D3"),
            ("D1", "U"),
            ("D2", "U"),
            ("D3", "U"),
        ],
        {
            "X": ("a", "b"),
            "D1": ("a", "b"),
            "D2": ("a", "b"),
            "D3": ("a", "b"),
            "U": ("p",),
        },
        {
            "X": "chance",
            "D1": "decision",
            "D2": "decision",
            "D3": "decision",
            "U": "utility",
        },
        True,
    ),
}


@pytest.mark.parametrize("case", BATTERY.values(), ids=list(BATTERY))
def test_matches_pyagrum_on_structural_battery(case):
    edges, states, kinds, expected = case
    snapshot = _pylimid(edges, states, kinds).snapshot()
    assert is_solvable(snapshot) is expected
    assert _pyagrum_is_solvable(_pyagrum(edges, states, kinds)) is expected


#: pyAgrum's ``isSolvable`` only tests decisions within a level of its partial
#: order. ``D1`` sits at a higher level but influences ``D2``'s utilities
#: directly and is not observed by ``D2``, so aGrUM admits the diagram and
#: returns a suboptimal policy — the exact-solution-ordering criterion
#: (Lauritzen and Nilsson, 2001) rejects it. Pinned here so an upstream fix
#: is noticed.
PYAGRUM_DIVERGENCE = {
    "higher_level_decision_influences_unobserved": _case(
        [("D1", "X"), ("X", "D2"), ("D1", "U"), ("D2", "U")],
        {
            "D1": ("a", "b"),
            "X": ("a", "b"),
            "D2": ("a", "b"),
            "U": ("p",),
        },
        {
            "D1": "decision",
            "X": "chance",
            "D2": "decision",
            "U": "utility",
        },
        False,
    ),
}


@pytest.mark.parametrize(
    "case", PYAGRUM_DIVERGENCE.values(), ids=list(PYAGRUM_DIVERGENCE)
)
def test_pyagrum_admits_what_the_paper_rejects(case):
    edges, states, kinds, expected = case
    snapshot = _pylimid(edges, states, kinds).snapshot()
    assert is_solvable(snapshot) is expected
    assert _pyagrum_is_solvable(_pyagrum(edges, states, kinds)) is True


def test_nested_order_solves_the_last_decision_first():
    snapshot = NestedTwoDecisions.pylimid().snapshot()
    assert solvability_order(snapshot) == ["D2", "D1"]


def test_three_decision_chain_order():
    edges, states, kinds, _ = BATTERY["three_decision_chain"]
    assert solvability_order(_pylimid(edges, states, kinds).snapshot()) == [
        "D3",
        "D2",
        "D1",
    ]


def test_non_solvable_has_no_order():
    edges, states, kinds, _ = BATTERY["shared_utility_no_ordering"]
    assert solvability_order(_pylimid(edges, states, kinds).snapshot()) is None
