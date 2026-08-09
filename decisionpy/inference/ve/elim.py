"""
Variable elimination — exact marginal inference on discrete Bayesian networks.

Graph-native, zero heavy dependencies.  Works directly against a
:class:`~decisionpy.graph.diagram.Snapshot`.

Algorithm (for a single query variable)
---------------------------------------
1. Build one :class:`~decisionpy.inference._factor.Factor` per node (its CPT).
2. Condition on observed variables by slicing factors.
3. Determine an elimination order (all variables except query + evidence).
4. For each variable *X* in order:
   - Multiply all factors that mention *X*.
   - Sum *X* out of the product.
   - Replace the consumed factors with the result.
5. Multiply and normalise the remaining factors (which now span only query
   variables).

Example:
-------
>>> from decisionpy.graph import ChanceNode, InfluenceDiagram
>>> from decisionpy.inference.ve import query
>>> import numpyro.distributions as dist
>>> d = InfluenceDiagram()
>>> d.add_node(
...     ChanceNode(
...         "rain",
...         states=("no", "yes"),
...         dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
...     )
... )
>>> d.add_node(
...     ChanceNode(
...         "wet_grass",
...         parents=("rain",),
...         states=("dry", "wet"),
...         dist=lambda rain: dist.Categorical(probs=cpt[rain]),
...     )
... )
>>> query(d.snapshot(), variables=["rain"])  # prior
{'rain': [0.8, 0.2]}
>>> query(d.snapshot(), variables=["rain"], observed={"wet_grass": 1})
{'rain': [0.457..., 0.542...]}
"""

from __future__ import annotations

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.inference._cpt import cardinalities, cpt
from decisionpy.inference._factor import Factor

__all__ = ["query"]


def query(
    snapshot: Snapshot,
    *,
    variables: list[str],
    observed: dict[str, int] | None = None,
) -> dict[str, list[float]]:
    """
    Compute the marginal posterior of *variables* given *observed* evidence.

    Requires a discrete-only Bayesian network (every node must have ``states``
    set).  All nodes must be ``CONSISTENT`` prior to snapshotting.

    Returns:
        A dict mapping each query variable name to its probability vector
        ``[P(v=0), P(v=1), ...]``.
    """
    observed = observed or {}
    _check_discrete(snapshot)
    card = cardinalities(snapshot)

    # 1. Build initial factor list (CPTs).
    #
    # VE is a chance-node engine (a discrete Bayesian network). ``_check_discrete``
    # above guarantees every node ``is_discrete``, which only chance nodes can be —
    # but the snapshot carries the ``Node`` base type, so narrow to ``ChanceNode``
    # for the chance-specific CPT logic. This is a type narrowing, not a behavior
    # change: a diagram with decisions / utilities routes to ``solve()``, not here.
    node_map: dict[str, ChanceNode] = {}
    for _name, node in snapshot.nodes:
        assert isinstance(node, ChanceNode)
        node_map[_name] = node
    factors: list[Factor] = []
    for _name, node in node_map.items():
        factors.append(cpt(node, card))

    # Condition: for each observed variable, slice its dimension in EVERY factor
    # that contains it.
    for obs_var, obs_val in observed.items():
        for i, f in enumerate(factors):
            if obs_var in f.variables:
                factors[i] = f.condition(obs_var, obs_val)

    # Drop factors that became empty (constant 1) after conditioning.
    factors = [f for f in factors if f.variables]

    # 2. Elimination order: all variables except query + evidence.
    protected = set(variables) | set(observed)
    order = [name for name, _ in snapshot.nodes if name not in protected]

    # 3. Eliminate.
    for var in order:
        relevant = [f for f in factors if var in f.variables]
        if not relevant:
            continue
        factors = [f for f in factors if f not in relevant]
        product = relevant[0]
        for f in relevant[1:]:
            product = product * f
        factors.append(product.marginal([v for v in product.variables if v != var]))

    # 4. Multiply remaining factors.
    if not factors:
        return {v: _point_mass(v, observed, node_map) for v in variables}
    joint = factors.pop(0)
    for f in factors:
        joint = joint * f
    joint.normalize()

    # 5. Marginals per query variable.
    result: dict[str, list[float]] = {}
    for v in variables:
        if v in observed:
            assert node_map[v].states is not None
            card = len(node_map[v].states)  # ty: ignore[invalid-argument-type]
            point = [0.0] * card
            point[observed[v]] = 1.0
            result[v] = point
        else:
            margin = joint.marginal([v])
            result[v] = list(margin.values)
    return result


# ---------------------------------------------------------------------------


def _check_discrete(snapshot: Snapshot) -> None:
    for _, node in snapshot.nodes:
        if not node.is_discrete:
            raise TypeError(
                f"Variable elimination requires all-discrete diagrams. "
                f"Node {node.name!r} has no ``states`` and appears continuous."
            )


def _point_mass(
    var: str,
    observed: dict[str, int],
    node_map: dict[str, ChanceNode],
) -> list[float]:
    """Return a point-mass vector for *var*: [0,...,1,...,0] at its observed value."""
    assert node_map[var].states is not None
    card = len(node_map[var].states)  # ty: ignore[invalid-argument-type]
    point = [0.0] * card
    if var in observed:
        point[observed[var]] = 1.0
    else:
        point[0] = 1.0  # uniform choice for prior over empty diagram
    return point
