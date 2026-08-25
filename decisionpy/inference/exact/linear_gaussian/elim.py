"""
Linear-Gaussian variable elimination — exact inference on all-continuous BNs.

The continuous counterpart of variable_elim: the same elimination schedule
over canonical-form Gaussian factors instead of discrete tables. Every
node is a Normal whose mean is linear in its continuous parents; each CPD
is probed into (intercept, slopes, scale) and converted to a GaussianFactor.

Algorithm (for a single query variable)
---------------------------------------
1. Build one GaussianFactor per node (its CPD in canonical form).
2. Condition on observed variables by clamping factors.
3. Eliminate every variable except query + evidence (Schur-complement
   each variable out of the product of the factors that mention it).
4. Multiply the remaining factors; a query variable's marginal is then a
   single-variable Gaussian, whose moments are read off the canonical form.

Example:
--------
>>> from decisionpy.graph import ChanceNode, InfluenceDiagram
>>> from decisionpy.inference.exact.linear_gaussian import query
>>> import numpyro.distributions as dist
>>> d = InfluenceDiagram()
>>> d.add_node(ChanceNode("x1", dist=lambda: dist.Normal(0.0, 1.0)))
>>> d.add_node(
...     ChanceNode(
...         "x2",
...         parents=("x1",),
...         dist=lambda x1: dist.Normal(2.0 + 1.5 * x1, 0.5),
...     )
... )
>>> query(d.snapshot(), variables=["x2"])  # prior: x2 ~ N(2, 2.5)
{'x2': (2.0, 2.5...)}
>>> query(d.snapshot(), variables=["x1"], observed={"x2": 0.4})
{'x1': (-0.96..., 0.1...)}
"""

from __future__ import annotations

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.inference.exact.linear_gaussian.gaussian_cpd import gaussian_cpd
from decisionpy.inference.exact.linear_gaussian.gaussian_factor import (
    GaussianFactor,
)

__all__ = ["query"]

#: Type of an LG query result: variable name -> (mean, variance).
LGResult = dict[str, tuple[float, float]]


def query(
    snapshot: Snapshot,
    *,
    variables: list[str],
    observed: dict[str, float] | None = None,
) -> LGResult:
    """
    Compute the marginal posterior of *variables* given *observed* evidence.

    Requires an all-continuous diagram whose every chance node is a Normal
    with a mean affine in its continuous parents. Utility nodes are
    ignored; decisions are not supported (bound them via the engine layer
    by clamping, as in the discrete engine).

    Args:
        snapshot: A validated snapshot of an all-continuous BN.
        variables: Names of the chance variables whose posteriors are
            requested.
        observed: Map from node name to its observed value.

    Returns:
        A dict mapping each query variable name to its ``(mean, variance)``
        posterior.

    Raises:
        TypeError: If a CPD is not a Normal.
        ValueError: If a CPD is not affine in its parents.
    """
    observed = observed or {}
    node_map: dict[str, ChanceNode] = {}
    for name, node in snapshot.nodes:
        if isinstance(node, ChanceNode):
            node_map[name] = node

    factors: list[GaussianFactor] = []
    for name, node in node_map.items():
        parents = list(node.parents)
        intercept, slopes, scale = gaussian_cpd(node)
        factors.append(GaussianFactor.from_cpd(name, parents, intercept, slopes, scale))

    # Condition: clamp the observed dimensions in every factor that has them.
    for i, f in enumerate(factors):
        relevant = {v: value for v, value in observed.items() if v in f.variables}
        if relevant:
            factors[i] = f.condition(relevant)

    # Elimination order: every variable except query + evidence.
    protected = set(variables) | set(observed)
    order = [name for name, _ in snapshot.nodes if name not in protected]

    for var in order:
        relevant = [f for f in factors if var in f.variables]
        if not relevant:
            continue
        factors = [f for f in factors if f not in relevant]
        product = relevant[0]
        for f in relevant[1:]:
            product = product * f
        factors.append(product.marginal([v for v in product.variables if v != var]))

    # Multiply the remaining factors; each query variable's marginal is a
    # single-variable Gaussian.
    if not factors:
        return {v: _point_mass(v, observed) for v in variables}
    joint = factors.pop(0)
    for f in factors:
        joint = joint * f

    result: LGResult = {}
    for v in variables:
        if v in observed:
            result[v] = _point_mass(v, observed)
        else:
            margin = joint.marginal([v])
            result[v] = (
                float(margin.info[0] / margin.precision[0, 0]),
                float(1.0 / margin.precision[0, 0]),
            )
    return result


def _point_mass(v: str, observed: dict[str, float]) -> tuple[float, float]:
    """Return the observed value as a zero-variance posterior."""
    return float(observed[v]), 0.0
