"""
Shared categorical-BN fixtures: a decisionpy diagram and a pyAgrum model.

Each fixture is defined once and builds the same CPTs in both libraries,
so the numpyro comparison test (test_compare_pyagrum.py) has a single
source of truth: the decisionpy diagram feeds infer(), and the pyAgrum
model's exact LazyPropagation inference is the reference answer.
"""

from __future__ import annotations

import inspect
from itertools import product

import jax.numpy as jnp
import numpyro.distributions as dist
import pyagrum as gum

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram

__all__ = [
    "TwoNodeRainWet",
    "ThreeNodeChain",
    "VStructure",
    "FourNodeAsia",
]


class BNFixture:
    """A Bayesian network definition usable by decisionpy and pyAgrum."""

    #: Edge list [(parent, child), ...]
    edges: list[tuple[str, str]]
    #: {name: states_tuple}
    states: dict[str, tuple[str, ...]]
    #: {name: (cpt_array, evidence_list)}
    #: CPT as a flat array in row-major order over (evidence..., var).
    #: evidence_list gives the order of parent dimensions.
    cpds: dict[str, tuple[list[float], list[str]]]

    def decisionpy(self) -> InfluenceDiagram:
        """Build a decisionpy InfluenceDiagram."""
        diag = InfluenceDiagram()
        # Build parent lookup from edges.
        children: dict[str, list[str]] = {}
        for p, c in self.edges:
            children.setdefault(c, []).append(p)

        for name, sts in self.states.items():
            parents = tuple(children.get(name, []))
            cpt_values, cpt_evidence = self.cpds[name]
            assert list(parents) == cpt_evidence, (
                f"CPT evidence order mismatch for {name}: "
                f"parents={parents}, cpt_evidence={cpt_evidence}"
            )

            # Determine the CPT shape: card_of(parents) + card_of(var).
            parent_cards = [len(self.states[p]) for p in parents]
            var_card = len(sts)
            total = 1
            for c in parent_cards:
                total *= c
            rows = int(total)

            # The flat list is [row0_val0, row0_val1, ..., row1_val0, ...]
            rows_data = []
            for r in range(rows):
                start = r * var_card
                rows_data.append(cpt_values[start : start + var_card])

            # Build a JAX/numpy array the dist lambda can index into.
            shape = parent_cards + [var_card]
            arr = (
                jnp.array(rows_data).reshape(shape)
                if parent_cards
                else jnp.array(rows_data[0])
            )

            if not parents:
                node = ChanceNode(
                    name=name,
                    states=sts,
                    dist=lambda arr=arr: dist.Categorical(probs=arr),
                )
            else:
                # Since CPT values are in [parents..., var] order and the lambda
                # receives parents as keyword args, index the array generically.
                # The signature is declared with one named parameter per parent
                # so the consistency gate can verify parent coverage.
                def make_dist(_arr, _parents):
                    def fn(**kwargs):
                        idx = tuple(kwargs[p] for p in _parents)
                        return dist.Categorical(probs=_arr[idx])

                    fn.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
                        [
                            inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY)
                            for p in _parents
                        ]
                    )
                    return fn

                node = ChanceNode(
                    name=name,
                    parents=parents,
                    states=sts,
                    dist=make_dist(arr, parents),
                )
            diag.add_node(node)

        return diag

    def pyagrum(self) -> gum.BayesNet:
        """Build a pyAgrum BayesNet with the same CPTs."""
        bn = gum.BayesNet()
        for name, sts in self.states.items():
            bn.add(gum.LabelizedVariable(name, name, list(sts)))
        for parent, child in self.edges:
            bn.addArc(bn.idFromName(parent), bn.idFromName(child))

        for name, sts in self.states.items():
            values, evidence = self.cpds[name]
            tbl = bn.cpt(name)
            parent_cards = [len(self.states[e]) for e in evidence]
            rows = _rows(values, len(sts))
            for i, assign in enumerate(product(*[range(c) for c in parent_cards])):
                tbl[dict(zip(evidence, assign, strict=True))] = rows[i]
        return bn

    def pyagrum_query(
        self,
        variables: list[str],
        observed: dict[str, int] | None = None,
    ) -> dict[str, list[float]]:
        """Run pyAgrum LazyPropagation; return (mean, std) pairs."""
        ie = gum.LazyPropagation(self.pyagrum())
        evidence = {}
        if observed:
            for var, val in observed.items():
                evidence[var] = self.states[var][val]
        if evidence:
            ie.setEvidence(evidence)
        ie.makeInference()
        result: dict[str, list[float]] = {}
        for var in variables:
            result[var] = [float(x) for x in ie.posterior(var).tolist()]
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows(values: list[float], var_card: int) -> list[list[float]]:
    """Split a flat value list into rows of *var_card* entries."""
    return [values[i : i + var_card] for i in range(0, len(values), var_card)]


# ---------------------------------------------------------------------------

TwoNodeRainWet = BNFixture()
TwoNodeRainWet.edges = [("rain", "wet_grass")]
TwoNodeRainWet.states = {
    "rain": ("no", "yes"),
    "wet_grass": ("dry", "wet"),
}
TwoNodeRainWet.cpds = {
    "rain": ([0.8, 0.2], []),
    "wet_grass": (
        [0.80, 0.20, 0.05, 0.95],  # rain=no, rain=yes (row-major)
        ["rain"],
    ),
}

# ---------------------------------------------------------------------------

ThreeNodeChain = BNFixture()
ThreeNodeChain.edges = [("a", "b"), ("b", "c")]
ThreeNodeChain.states = {
    "a": ("a0", "a1"),
    "b": ("b0", "b1"),
    "c": ("c0", "c1"),
}
ThreeNodeChain.cpds = {
    "a": ([0.4, 0.6], []),
    "b": (
        # P(b|a): a=0 -> [0.7, 0.3]; a=1 -> [0.2, 0.8]
        [0.7, 0.3, 0.2, 0.8],
        ["a"],
    ),
    "c": (
        # P(c|b): b=0 -> [0.9, 0.1]; b=1 -> [0.5, 0.5]
        [0.9, 0.1, 0.5, 0.5],
        ["b"],
    ),
}

# ---------------------------------------------------------------------------

VStructure = BNFixture()
VStructure.edges = [("a", "c"), ("b", "c")]
VStructure.states = {
    "a": ("a0", "a1"),
    "b": ("b0", "b1"),
    "c": ("c0", "c1"),
}
VStructure.cpds = {
    "a": ([0.5, 0.5], []),
    "b": ([0.6, 0.4], []),
    "c": (
        # P(c|a,b): row-major over product(a, b)
        # a=0,b=0; a=0,b=1; a=1,b=0; a=1,b=1
        [0.95, 0.05, 0.30, 0.70, 0.20, 0.80, 0.01, 0.99],
        ["a", "b"],
    ),
}

# ---------------------------------------------------------------------------
# Toy "Asia-like" 4-node BN: smoking → cancer ← pollution
#                                ↓
#                              xray

FourNodeAsia = BNFixture()
FourNodeAsia.edges = [
    ("smoking", "cancer"),
    ("pollution", "cancer"),
    ("cancer", "xray"),
]
FourNodeAsia.states = {
    "smoking": ("no", "yes"),
    "pollution": ("low", "high"),
    "cancer": ("absent", "present"),
    "xray": ("neg", "pos"),
}
FourNodeAsia.cpds = {
    "smoking": ([0.7, 0.3], []),
    "pollution": ([0.9, 0.1], []),
    "cancer": (
        # P(cancer | smoking, pollution)
        # smoking=no, pollution=low  → [0.99, 0.01]
        # smoking=no, pollution=high → [0.95, 0.05]
        # smoking=yes, pollution=low → [0.90, 0.10]
        # smoking=yes, pollution=high→ [0.50, 0.50]
        [0.99, 0.01, 0.95, 0.05, 0.90, 0.10, 0.50, 0.50],
        ["smoking", "pollution"],
    ),
    "xray": (
        # P(xray | cancer)
        # cancer=absent  → [0.80, 0.20]
        # cancer=present → [0.10, 0.90]
        [0.80, 0.20, 0.10, 0.90],
        ["cancer"],
    ),
}
