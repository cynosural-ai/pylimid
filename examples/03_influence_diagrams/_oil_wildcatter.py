"""
The Oil Wildcatter problem (Howard, classic decision analysis textbook example).

A shared model definition: one spec (edges, states, kinds, CPTs, utilities)
builds both the decisionpy influence diagram and the pyAgrum one, so the
exact LIMID solution (pyAgrum) and the NumPyro intervention scan
(decisionpy) can be compared head to head. The numbers come from pyAgrum's
own example file ``res/OilWildcatter.bgum`` (aGrUM repository).

The plot: an oil deposit may be Dry, Wet, or Soaking. Before deciding to
drill, we may run an expensive test whose report (closed / open / diffuse)
is informative about the deposit. Drilling pays off per deposit type;
testing costs 10.

- OilContents (chance): Dry / Wet / Soaking.
- Testing (decision): Yes / No — run the test or not (no information set).
- TestResult (chance): closed / open / diffuse — the report, depends on
  the deposit and on whether the test was run (uniform when not run).
- Drilling (decision): Yes / No — observes TestResult and Testing.
- Cost (utility): -10 when testing.
- Reward (utility): -70 / 50 / 200 when drilling on Dry / Wet / Soaking,
  0 when not drilling.

The exact optimum (pyAgrum): run the test, drill unless the report is
"diffuse"; MEU = 22.5.
"""

from __future__ import annotations

import inspect
from itertools import product

import jax.numpy as jnp
import numpyro.distributions as dist
import pyagrum as gum

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

__all__ = ["decisionpy_diagram", "pyagrum_diagram", "pyagrum_solution", "spec"]

#: Shared spec: ``(edges, states, kinds, cpds, utilities)``. ``cpds`` maps a
#: chance node to ``(flat_cpt, parent_order)`` — row-major over the parents
#: followed by the node itself. ``utilities`` maps a utility node to
#: ``(flat_values, parent_order)`` — row-major over the parents.
spec: tuple = (
    [
        ("OilContents", "TestResult"),
        ("Testing", "TestResult"),
        ("Testing", "Drilling"),
        ("TestResult", "Drilling"),
        ("Testing", "Cost"),
        ("OilContents", "Reward"),
        ("Drilling", "Reward"),
    ],
    {
        "OilContents": ("Dry", "Wet", "Soaking"),
        "Testing": ("No", "Yes"),
        "TestResult": ("closed", "open", "diffuse"),
        "Drilling": ("No", "Yes"),
        "Cost": ("pay",),
        "Reward": ("pay",),
    },
    {
        "OilContents": "chance",
        "Testing": "decision",
        "TestResult": "chance",
        "Drilling": "decision",
        "Cost": "utility",
        "Reward": "utility",
    },
    {
        "OilContents": ([0.5, 0.3, 0.2], []),
        # P(TestResult | OilContents, Testing): the report is informative
        # only when the test is actually run; without it, it is uniform.
        "TestResult": (
            [
                1 / 3,
                1 / 3,
                1 / 3,  # Dry, No
                0.10,
                0.30,
                0.60,  # Dry, Yes
                1 / 3,
                1 / 3,
                1 / 3,  # Wet, No
                0.30,
                0.40,
                0.30,  # Wet, Yes
                1 / 3,
                1 / 3,
                1 / 3,  # Soaking, No
                0.50,
                0.40,
                0.10,  # Soaking, Yes
            ],
            ["OilContents", "Testing"],
        ),
    },
    {
        # Cost = -10 when testing, 0 otherwise.
        "Cost": ([0.0, -10.0], ["Testing"]),
        # Reward = -70 / 50 / 200 when drilling on Dry / Wet / Soaking, else 0.
        "Reward": (
            [0.0, -70.0, 0.0, 50.0, 0.0, 200.0],
            ["OilContents", "Drilling"],
        ),
    },
)


def decisionpy_diagram() -> InfluenceDiagram:
    """Build the decisionpy influence diagram from the shared spec."""
    edges, states, kinds, cpds, utilities = spec
    diag = InfluenceDiagram()
    for name, sts in states.items():
        parents = tuple(p for p, c in edges if c == name)
        kind = kinds[name]
        if kind == "chance":
            values, evidence = cpds[name]
            assert list(parents) == evidence
            arr = jnp.array(values).reshape(
                [len(states[e]) for e in evidence] + [len(sts)]
            )
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
            values, evidence = utilities[name]
            assert list(parents) == evidence
            arr = jnp.array(values).reshape([len(states[e]) for e in evidence])
            diag.add_node(
                UtilityNode(
                    name=name,
                    parents=parents,
                    values=_values_factory(arr, evidence),
                )
            )
    return diag


def pyagrum_diagram() -> gum.InfluenceDiagram:  # ty: ignore[possibly-missing-attribute]
    """Build the pyAgrum influence diagram with identical CPTs and utilities."""
    edges, states, kinds, cpds, utilities = spec
    g = gum.InfluenceDiagram()  # ty: ignore[possibly-missing-attribute]
    for name, sts in states.items():
        var = gum.LabelizedVariable(name, name, list(sts))
        kind = kinds[name]
        if kind == "chance":
            g.addChanceNode(var)
        elif kind == "decision":
            g.addDecisionNode(var)
        else:
            g.addUtilityNode(var)
    for parent, child in edges:
        g.addArc(g.idFromName(parent), g.idFromName(child))

    for name, sts in states.items():
        kind = kinds[name]
        if kind == "chance":
            values, evidence = cpds[name]
            parent_cards = [len(states[e]) for e in evidence]
            rows = [values[i : i + len(sts)] for i in range(0, len(values), len(sts))]
            for i, assign in enumerate(product(*[range(c) for c in parent_cards])):
                g.cpt(name)[dict(zip(evidence, assign, strict=True))] = rows[i]
        elif kind == "utility":
            values, evidence = utilities[name]
            parent_cards = [len(states[e]) for e in evidence]
            for i, assign in enumerate(product(*[range(c) for c in parent_cards])):
                g.utility(name)[dict(zip(evidence, assign, strict=True))] = values[i]
    return g


def pyagrum_solution() -> tuple[float, dict[str, dict[tuple[int, ...], int]]]:
    """
    Solve the diagram with pyAgrum's exact LIMID solver.

    Returns:
        ``(meu, policy)`` — the mean expected utility and the per-decision
        optimal policy in decisionpy's format (decision name to
        ``{info_set_assignment: action}``).
    """
    ie = gum.ShaferShenoyLIMIDInference(pyagrum_diagram())  # ty: ignore[possibly-missing-attribute]
    ie.makeInference()
    meu = float(ie.MEU()["mean"])

    _, states, kinds, _, _ = spec
    policy: dict[str, dict[tuple[int, ...], int]] = {}
    for name, sts in states.items():
        if kinds[name] != "decision":
            continue
        od = ie.optimalDecision(name)
        info_order = tuple(p for p, c in spec[0] if c == name)
        info_cards = [len(states[p]) for p in info_order]
        sub: dict[tuple[int, ...], int] = {}
        for assign in product(*[range(c) for c in info_cards]):
            entry = dict(zip(info_order, assign, strict=True))
            for a in range(len(sts)):
                if float(od[{**entry, name: a}]) >= 0.999999:
                    sub[assign] = a
                    break
        policy[name] = sub
    return meu, policy


# --- builders for the decisionpy side ----------------------------------------


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
    can verify parent coverage.
    """

    def fn(**kwargs):
        idx = tuple(kwargs[p] for p in evidence)
        return float(arr[idx])

    return _declare_signature(fn, evidence)


def _declare_signature(fn, names: list[str]):
    """Attach a named-parameter signature to a ``**kwargs``-style factory."""
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    return fn
