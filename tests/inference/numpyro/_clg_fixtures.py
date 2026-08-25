"""
Shared linear-Gaussian fixtures: a decisionpy diagram and a pyAgrum CLG net.

Each fixture is defined once by its CPD parameters (intercept, per-parent
slopes, scale) and builds a decisionpy InfluenceDiagram whose dist
callables reproduce those parameters, and a pyAgrum pyagrum.clg network
with the same mu/sigma/coef. The pyAgrum network's exact posteriors are
the reference answer the numpyro MCMC comparison tests against.
"""

from __future__ import annotations

import inspect

import numpyro.distributions as dist
import pyagrum.clg as gclg

from decisionpy.graph import ChanceNode
from decisionpy.graph.diagram import InfluenceDiagram

__all__ = ["SingleRoot", "Chain", "VStructure", "LongChain"]


class LGFixture:
    """
    A linear-Gaussian BN definition, in topological order.

    Attributes:
        cpds: ``{name: (intercept, {parent: slope}, scale)}`` — insertion
            order is the topological order (parents before children).
    """

    cpds: dict[str, tuple[float, dict[str, float], float]]

    def decisionpy(self) -> InfluenceDiagram:
        """Build a decisionpy InfluenceDiagram with the same CPDs."""
        diag = InfluenceDiagram()
        for name, (intercept, slopes, scale) in self.cpds.items():
            parents = tuple(slopes)

            def make_dist(intercept, slopes, scale, parents):
                def fn(**kwargs):
                    loc = intercept + sum(
                        slope * kwargs[parent] for parent, slope in slopes.items()
                    )
                    return dist.Normal(loc, scale)

                fn.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
                    [
                        inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY)
                        for p in parents
                    ]
                )
                return fn

            diag.add_node(
                ChanceNode(
                    name=name,
                    parents=parents,
                    dist=make_dist(intercept, slopes, scale, parents),
                )
            )
        return diag

    def pyagrum(self) -> gclg.CLG:
        """Build a pyAgrum CLG net with the same mu/sigma/coef."""
        net = gclg.CLG()
        for name, (intercept, _slopes, scale) in self.cpds.items():
            net.add(gclg.GaussianVariable(name, mu=intercept, sigma=scale))
        for name, (_intercept, slopes, _scale) in self.cpds.items():
            for parent, coef in slopes.items():
                net.addArc(parent, name, coef=coef)
        return net

    def pyagrum_query(
        self,
        variables: list[str],
        observed: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Run pyAgrum CLG variable elimination; return (mean, variance)."""
        ie = gclg.CLGVariableElimination(self.pyagrum())
        if observed:
            ie.updateEvidence(observed)
        result: dict[str, tuple[float, float]] = {}
        for var in variables:
            post = ie.posterior(var)
            result[var] = (float(post.mu()), float(post.sigma() ** 2))
        return result


# ---------------------------------------------------------------------------
# Fixtures (topological order)
# ---------------------------------------------------------------------------

SingleRoot = LGFixture()
SingleRoot.cpds = {
    "x": (3.0, {}, 0.5),
}

# ---------------------------------------------------------------------------

Chain = LGFixture()
Chain.cpds = {
    "x1": (0.0, {}, 1.0),
    "x2": (2.0, {"x1": 1.5}, 0.5),
    "x3": (-1.0, {"x2": -0.7}, 0.3),
}

# ---------------------------------------------------------------------------

VStructure = LGFixture()
VStructure.cpds = {
    "x1": (0.0, {}, 1.0),
    "x2": (1.0, {}, 2.0),
    "y": (0.5, {"x1": 1.0, "x2": -0.5}, 0.25),
}

# ---------------------------------------------------------------------------

LongChain = LGFixture()
LongChain.cpds = {
    "a": (0.0, {}, 1.0),
    "b": (0.5, {"a": 0.8}, 0.6),
    "c": (-1.0, {"b": 1.2}, 0.4),
    "d": (2.0, {"c": -0.9}, 0.7),
}
