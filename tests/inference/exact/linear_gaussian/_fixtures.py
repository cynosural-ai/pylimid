"""
Shared linear-Gaussian fixtures: decisionpy diagram, pyAgrum CLG net, MVN oracle.

Each fixture is defined once by its CPD parameters (intercept, per-parent
slopes, scale) and builds:

- a decisionpy InfluenceDiagram whose dist callables reproduce those
  parameters;
- a pyAgrum pyagrum.clg network with the same mu/sigma/coef;
- the exact joint Gaussian computed in float64 from the same parameters,
  whose conditionals are the reference answer.

The oracle is independent of the engine's factor algebra: it builds the
full joint (I-B)^-1 a, (I-B)^-1 diag(sigma^2) (I-B)^-T and conditions by
the closed-form Schur complement, so any discrepancy against the engine
points at the engine, not at the shared math.
"""

from __future__ import annotations

import inspect

import numpy as np
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

    def oracle_query(
        self,
        variables: list[str],
        observed: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Exact conditional of the joint Gaussian, computed in float64.

        This is the reference implementation: it assembles the full joint
        from the CPD parameters and conditions by the closed-form Schur
        complement, never touching the engine's factor representation.
        """
        names = list(self.cpds)
        pos = {name: i for i, name in enumerate(names)}
        n = len(names)
        b_matrix = np.zeros((n, n))
        intercepts = np.zeros(n)
        scales = np.zeros(n)
        for name, (a, slopes, scale) in self.cpds.items():
            i = pos[name]
            intercepts[i] = a
            scales[i] = scale
            for parent, slope in slopes.items():
                b_matrix[i, pos[parent]] = slope

        a_matrix = np.linalg.inv(np.eye(n) - b_matrix)
        mu = a_matrix @ intercepts
        covariance = a_matrix @ np.diag(scales**2) @ a_matrix.T

        observed = observed or {}
        result: dict[str, tuple[float, float]] = {}
        for v in variables:
            if v in observed:
                result[v] = (float(observed[v]), 0.0)
        query_vars = [v for v in variables if v not in observed]
        if not query_vars:
            return result

        q_idx = [pos[v] for v in query_vars]
        e_idx = [pos[v] for v in observed]
        mu_q = mu[q_idx]
        sigma_q = covariance[np.ix_(q_idx, q_idx)]
        if e_idx:
            mu_e = mu[e_idx]
            sigma_qe = covariance[np.ix_(q_idx, e_idx)]
            sigma_ee = covariance[np.ix_(e_idx, e_idx)]
            e_values = np.array([observed[v] for v in observed])
            coupling = sigma_qe @ np.linalg.inv(sigma_ee)
            mu_q = mu_q + coupling @ (e_values - mu_e)
            sigma_q = sigma_q - coupling @ sigma_qe.T
        for k, v in enumerate(query_vars):
            result[v] = (float(mu_q[k]), float(sigma_q[k, k]))
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
