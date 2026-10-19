"""Shared model factories that produce both a pylimid and a pyAgrum LIMID."""

from __future__ import annotations

import inspect
import math
from itertools import product

import jax.numpy as jnp
import numpyro.distributions as dist
import pyagrum as gum

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

__all__ = [
    "SingleDecisionTreat",
    "TwoDecisionsIndependent",
    "EmptyInfoDecision",
    "IrrelevantInfoDecision",
    "NestedTwoDecisions",
    "ThreeStateClimate",
]


class IDFixture:
    """
    An influence-diagram definition usable by both pylimid and pyAgrum.

    Attributes:
        edges: Edge list ``[(parent, child), ...]``.
        states: ``{name: states}`` for chance and decision nodes (utility
            nodes carry a single placeholder label).
        kinds: ``{name: "chance" | "decision" | "utility"}``.
        cpds: ``{chance_name: (flat_cpt, parent_order)}`` — row-major over
            ``parent_order + [var]``.
        utilities: ``{utility_name: (flat_values, parent_order)}`` — row-major
            over the parents.
    """

    edges: list[tuple[str, str]]
    states: dict[str, tuple[str, ...]]
    kinds: dict[str, str]
    cpds: dict[str, tuple[list[float], list[str]]]
    utilities: dict[str, tuple[list[float], list[str]]]

    # -- pylimid ---------------------------------------------------------

    def pylimid(self) -> InfluenceDiagram:
        """Build the pylimid InfluenceDiagram."""
        diag = InfluenceDiagram()
        for name, sts in self.states.items():
            parents = _parents_of(self.edges, name)
            kind = self.kinds[name]
            if kind == "chance":
                values, evidence = self.cpds[name]
                assert list(parents) == evidence, (
                    f"CPT evidence order mismatch for {name}: "
                    f"parents={parents}, cpt_evidence={evidence}"
                )
                arr = _cpt_array(values, evidence, self.states, len(sts))
                diag.add_node(
                    ChanceNode(
                        name=name,
                        parents=parents,
                        states=sts,
                        dist=_dist_factory(arr, evidence),
                    )
                )
            elif kind == "decision":
                diag.add_node(DecisionNode(name=name, parents=parents, states=sts))
            else:
                values, evidence = self.utilities[name]
                assert list(parents) == evidence, (
                    f"Utility evidence order mismatch for {name}: "
                    f"parents={parents}, utility_evidence={evidence}"
                )
                arr = _parent_array(values, evidence, self.states)
                diag.add_node(
                    UtilityNode(
                        name=name,
                        parents=parents,
                        values=_values_factory(arr, evidence),
                    )
                )
        return diag

    # -- pyAgrum ------------------------------------------------------------

    def pyagrum(self) -> gum.InfluenceDiagram:  # ty: ignore[possibly-missing-attribute]
        """Build a pyAgrum InfluenceDiagram with identical CPTs and utilities."""
        g = gum.InfluenceDiagram()  # ty: ignore[possibly-missing-attribute]
        for name, sts in self.states.items():
            var = gum.LabelizedVariable(name, name, list(sts))
            kind = self.kinds[name]
            if kind == "chance":
                g.addChanceNode(var)
            elif kind == "decision":
                g.addDecisionNode(var)
            else:
                g.addUtilityNode(var)
        for parent, child in self.edges:
            g.addArc(g.idFromName(parent), g.idFromName(child))

        for name, sts in self.states.items():
            kind = self.kinds[name]
            if kind == "chance":
                values, evidence = self.cpds[name]
                tbl = g.cpt(name)
                parent_cards = [len(self.states[e]) for e in evidence]
                rows = _rows(values, len(sts))
                for i, assign in enumerate(product(*[range(c) for c in parent_cards])):
                    tbl[dict(zip(evidence, assign, strict=True))] = rows[i]
            elif kind == "utility":
                values, evidence = self.utilities[name]
                tbl = g.utility(name)
                parent_cards = [len(self.states[e]) for e in evidence]
                for i, assign in enumerate(product(*[range(c) for c in parent_cards])):
                    tbl[dict(zip(evidence, assign, strict=True))] = values[i]
        return g

    def pyagrum_solution(self) -> tuple[float, dict[str, dict[tuple[int, ...], int]]]:
        """
        Solve the diagram with pyAgrum's exact LIMID solver.

        Returns:
            A ``(meu, policy)`` pair: the mean expected utility and the
            per-decision optimal policy in pylimid's format — decision name
            to ``{info_set_assignment: action}``.
        """
        ie = gum.ShaferShenoyLIMIDInference(  # ty: ignore[possibly-missing-attribute]
            self.pyagrum()
        )
        ie.makeInference()
        meu = float(ie.MEU()["mean"])

        policy: dict[str, dict[tuple[int, ...], int]] = {}
        for name, sts in self.states.items():
            if self.kinds[name] != "decision":
                continue
            od = ie.optimalDecision(name)
            info_order = _parents_of(self.edges, name)
            info_cards = [len(self.states[p]) for p in info_order]
            sub: dict[tuple[int, ...], int] = {}
            for assign in product(*[range(c) for c in info_cards]):
                entry = dict(zip(info_order, assign, strict=True))
                for a in range(len(sts)):
                    if math.isclose(float(od[{**entry, name: a}]), 1.0, abs_tol=1e-6):
                        sub[assign] = a
                        break
            policy[name] = sub
        return meu, policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parents_of(edges: list[tuple[str, str]], name: str) -> tuple[str, ...]:
    """Names of the parents of *name* (its information set, if a decision)."""
    return tuple(p for p, c in edges if c == name)


def _cpt_array(
    values: list[float],
    evidence: list[str],
    states: dict[str, tuple[str, ...]],
    var_card: int,
):
    """Reshape a flat row-major value list into an array over (evidence..., var)."""
    parent_cards = [len(states[e]) for e in evidence]
    return jnp.array(values).reshape(parent_cards + [var_card])


def _parent_array(
    values: list[float],
    evidence: list[str],
    states: dict[str, tuple[str, ...]],
):
    """Reshape a flat row-major value list into an array over the parents only."""
    parent_cards = [len(states[e]) for e in evidence]
    return jnp.array(values).reshape(parent_cards)


def _rows(values: list[float], var_card: int) -> list[list[float]]:
    """Split a flat value list into rows of *var_card* entries."""
    return [values[i : i + var_card] for i in range(0, len(values), var_card)]


def _dist_factory(arr, evidence: list[str]):
    """
    A chance-node dist factory indexing *arr* by parent name.

    The body resolves parents through ``kwargs``; the signature is declared
    with one named parameter per evidence variable so the consistency gate
    can verify parent coverage.
    """

    def fn(**kwargs):
        idx = tuple(kwargs[p] for p in evidence)
        return dist.Categorical(probs=arr[idx])

    return _declare_signature(fn, evidence)


def _values_factory(arr, evidence: list[str]):
    """
    A utility-node values factory indexing *arr* by parent name.

    The body resolves parents through ``kwargs``; the signature is declared
    with one named parameter per evidence variable so the consistency gate
    can verify parent coverage. Returns a JAX scalar so the callable is
    traceable inside the batched scan's vmapped evaluation.
    """

    def fn(**kwargs):
        idx = tuple(kwargs[p] for p in evidence)
        return arr[idx]

    return _declare_signature(fn, evidence)


def _declare_signature(fn, names: list[str]):
    """Attach a named-parameter signature to a ``**kwargs``-style factory."""
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    return fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SingleDecisionTreat = IDFixture()
SingleDecisionTreat.edges = [
    ("disease", "treat"),
    ("disease", "recovery"),
    ("treat", "recovery"),
    ("recovery", "utility"),
    ("treat", "utility"),
]
SingleDecisionTreat.states = {
    "disease": ("healthy", "sick"),
    "treat": ("no", "yes"),
    "recovery": ("no", "yes"),
    "utility": ("pay",),
}
SingleDecisionTreat.kinds = {
    "disease": "chance",
    "treat": "decision",
    "recovery": "chance",
    "utility": "utility",
}
SingleDecisionTreat.cpds = {
    "disease": ([0.6, 0.4], []),
    "recovery": (
        # P(recovery | disease, treat): healthy→no/yes, sick→no/yes
        [0.10, 0.90, 0.08, 0.92, 0.60, 0.40, 0.20, 0.80],
        ["disease", "treat"],
    ),
}
SingleDecisionTreat.utilities = {
    # U = 100·recovery − 20·treat
    "utility": ([0.0, -20.0, 100.0, 80.0], ["recovery", "treat"]),
}

# ---------------------------------------------------------------------------

TwoDecisionsIndependent = IDFixture()
TwoDecisionsIndependent.edges = [
    ("s1", "d1"),
    ("s1", "o1"),
    ("d1", "o1"),
    ("o1", "u1"),
    ("d1", "u1"),
    ("s2", "d2"),
    ("d2", "o2"),
    ("o2", "u2"),
    ("d2", "u2"),
]
TwoDecisionsIndependent.states = {
    "s1": ("a", "b"),
    "d1": ("a", "b"),
    "o1": ("no", "yes"),
    "u1": ("pay",),
    "s2": ("a", "b"),
    "d2": ("a", "b"),
    "o2": ("no", "yes"),
    "u2": ("pay",),
}
TwoDecisionsIndependent.kinds = {
    "s1": "chance",
    "d1": "decision",
    "o1": "chance",
    "u1": "utility",
    "s2": "chance",
    "d2": "decision",
    "o2": "chance",
    "u2": "utility",
}
TwoDecisionsIndependent.cpds = {
    "s1": ([0.6, 0.4], []),
    "o1": (
        # P(o1 | s1, d1)
        [0.9, 0.1, 0.5, 0.5, 0.2, 0.8, 0.1, 0.9],
        ["s1", "d1"],
    ),
    "s2": ([0.5, 0.5], []),
    "o2": (
        # P(o2 | d2)
        [0.6, 0.4, 0.3, 0.7],
        ["d2"],
    ),
}
TwoDecisionsIndependent.utilities = {
    "u1": ([0.0, -20.0, 100.0, 80.0], ["o1", "d1"]),  # 100·o1 − 20·d1
    "u2": ([0.0, -40.0, 100.0, 60.0], ["o2", "d2"]),  # 100·o2 − 40·d2
}

# ---------------------------------------------------------------------------

EmptyInfoDecision = IDFixture()
EmptyInfoDecision.edges = [
    ("act", "gain"),
    ("gain", "utility"),
    ("act", "utility"),
]
EmptyInfoDecision.states = {
    "act": ("no", "yes"),
    "gain": ("no", "yes"),
    "utility": ("pay",),
}
EmptyInfoDecision.kinds = {
    "act": "decision",
    "gain": "chance",
    "utility": "utility",
}
EmptyInfoDecision.cpds = {
    "gain": ([0.5, 0.5, 0.1, 0.9], ["act"]),
}
EmptyInfoDecision.utilities = {
    "utility": ([0.0, -30.0, 100.0, 70.0], ["gain", "act"]),  # 100·gain − 30·act
}

# ---------------------------------------------------------------------------

IrrelevantInfoDecision = IDFixture()
IrrelevantInfoDecision.edges = [
    ("y", "act"),
    ("act", "utility"),
]
IrrelevantInfoDecision.states = {
    "y": ("a", "b"),
    "act": ("no", "yes"),
    "utility": ("pay",),
}
IrrelevantInfoDecision.kinds = {
    "y": "chance",
    "act": "decision",
    "utility": "utility",
}
IrrelevantInfoDecision.cpds = {
    "y": ([0.5, 0.5], []),
}
IrrelevantInfoDecision.utilities = {
    "utility": ([100.0, 40.0], ["act"]),  # 100 − 60·act
}

# ---------------------------------------------------------------------------

NestedTwoDecisions = IDFixture()
NestedTwoDecisions.edges = [
    ("X", "D1"),
    ("X", "D2"),
    ("D1", "D2"),
    ("X", "Y"),
    ("D1", "Y"),
    ("D2", "Y"),
    ("Y", "U"),
    ("D1", "U"),
    ("D2", "U"),
]
NestedTwoDecisions.states = {
    "X": ("x0", "x1"),
    "D1": ("no", "yes"),
    "D2": ("no", "yes"),
    "Y": ("no", "yes"),
    "U": ("pay",),
}
NestedTwoDecisions.kinds = {
    "X": "chance",
    "D1": "decision",
    "D2": "decision",
    "Y": "chance",
    "U": "utility",
}
NestedTwoDecisions.cpds = {
    "X": ([0.5, 0.5], []),
    "Y": (
        # P(Y | X, D1, D2)
        [
            0.90,
            0.10,  # x0, no,  no
            0.70,
            0.30,  # x0, no,  yes
            0.60,
            0.40,  # x0, yes, no
            0.20,
            0.80,  # x0, yes, yes
            0.50,
            0.50,  # x1, no,  no
            0.30,
            0.70,  # x1, no,  yes
            0.20,
            0.80,  # x1, yes, no
            0.05,
            0.95,  # x1, yes, yes
        ],
        ["X", "D1", "D2"],
    ),
}
NestedTwoDecisions.utilities = {
    # U = 100·Y − 30·D1 − 10·D2
    "U": (
        [0.0, -10.0, -30.0, -40.0, 100.0, 90.0, 70.0, 60.0],
        ["Y", "D1", "D2"],
    ),
}

# ---------------------------------------------------------------------------

ThreeStateClimate = IDFixture()
ThreeStateClimate.edges = [
    ("weather", "shelter"),
    ("weather", "outcome"),
    ("shelter", "outcome"),
    ("outcome", "utility"),
    ("shelter", "utility"),
]
ThreeStateClimate.states = {
    "weather": ("sunny", "cloudy", "rainy"),
    "shelter": ("none", "cover"),
    "outcome": ("fine", "damaged", "ruined"),
    "utility": ("pay",),
}
ThreeStateClimate.kinds = {
    "weather": "chance",
    "shelter": "decision",
    "outcome": "chance",
    "utility": "utility",
}
ThreeStateClimate.cpds = {
    "weather": ([0.5, 0.3, 0.2], []),
    "outcome": (
        # P(outcome | weather, shelter)
        [
            0.90,
            0.08,
            0.02,  # sunny, none
            0.95,
            0.04,
            0.01,  # sunny, cover
            0.60,
            0.30,
            0.10,  # cloudy, none
            0.80,
            0.15,
            0.05,  # cloudy, cover
            0.10,
            0.40,
            0.50,  # rainy, none
            0.60,
            0.35,
            0.05,  # rainy, cover
        ],
        ["weather", "shelter"],
    ),
}
ThreeStateClimate.utilities = {
    # U = 100/20/−50 by outcome, minus 15 when sheltering
    "utility": (
        [100.0, 85.0, 20.0, 5.0, -50.0, -65.0],
        ["outcome", "shelter"],
    ),
}
