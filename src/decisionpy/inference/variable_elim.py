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

Example
-------

>>> from decisionpy.graph import ChanceNode, InfluenceDiagram
>>> from decisionpy.inference.variable_elim import query
>>> import numpyro.distributions as dist
>>> d = InfluenceDiagram()
>>> d.add_node(ChanceNode("rain",  states=("no","yes"),
...            dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2]))))
>>> d.add_node(ChanceNode("wet_grass", parents=("rain",), states=("dry","wet"),
...            dist=lambda rain: dist.Categorical(probs=cpt[rain])))
>>> query(d.snapshot(), variables=["rain"])  # prior
{'rain': [0.8, 0.2]}
>>> query(d.snapshot(), variables=["rain"], observed={"wet_grass": 1})
{'rain': [0.457..., 0.542...]}
"""

from __future__ import annotations

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.diagram import Snapshot
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

    :returns: A dict mapping each query variable name to its probability
              vector ``[P(v=0), P(v=1), ...]``.
    """
    observed = observed or {}
    _check_discrete(snapshot)

    # 1. Build initial factor list (CPTs).
    node_map = {name: node for name, node in snapshot.nodes}
    factors: list[Factor] = []
    for name, node in snapshot.nodes:
        factors.append(_cpt(node, node_map))

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
        return {
            v: _point_mass(v, observed, node_map)
            for v in variables
        }
    joint = factors.pop(0)
    for f in factors:
        joint = joint * f
    joint.normalize()

    # 5. Marginals per query variable.
    result: dict[str, list[float]] = {}
    for v in variables:
        if v in observed:
            card = len(node_map[v].states)  # type: ignore[arg-type]
            point = [0.0] * card
            point[observed[v]] = 1.0
            result[v] = point
        else:
            margin = joint.marginal([v])
            result[v] = list(margin.values)
    return result


# ---------------------------------------------------------------------------


def _cpt(node: ChanceNode, node_map: dict[str, ChanceNode]) -> Factor:
    """Build the CPT factor P(*node* | parents(*node*))."""
    parent_vars = list(node.parents)
    parent_cards = [len(node_map[p].states) for p in parent_vars]  # type: ignore[arg-type]
    node_card = len(node.states)  # type: ignore[arg-type]
    all_vars = parent_vars + [node.name]
    all_cards = parent_cards + [node_card]

    from itertools import product

    values: list[float] = []
    for parent_vals in product(*[range(c) for c in parent_cards]):
        kwargs = dict(zip(node.parents, parent_vals))
        prob_vec = _extract_probs(node, kwargs, node_card)
        if prob_vec is None:
            raise RuntimeError(
                f"Could not extract probability vector from node "
                f"{node.name!r}. Make sure its ``dist`` returns a "
                f"Categorical or a distribution with a ``probs`` attribute."
            )
        values.extend(prob_vec)

    return Factor(variables=all_vars, card=tuple(all_cards), values=values)


def _extract_probs(
    node: ChanceNode,
    parent_kwargs: dict[str, object],
    card: int,
) -> list[float] | None:
    """Call ``node.dist(**parent_kwargs)`` and extract probability vector."""
    dist = node.dist(**parent_kwargs)
    probs = getattr(dist, "probs", None)
    if probs is not None:
        return [float(p) for p in probs]
    # Fallback: materialise via log_prob for each outcome.
    try:
        log_probs = [dist.log_prob(i) for i in range(card)]
        import math

        return [math.exp(float(lp)) for lp in log_probs]
    except Exception:
        return None


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
    """Return a point-mass vector for *var*: [0,...,1,...,0] at its (possibly observed) value."""
    card = len(node_map[var].states)  # type: ignore[arg-type]
    point = [0.0] * card
    if var in observed:
        point[observed[var]] = 1.0
    else:
        point[0] = 1.0  # uniform choice for prior over empty diagram
    return point
