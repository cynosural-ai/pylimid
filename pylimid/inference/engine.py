"""
Unified inference entry-point.

::

    from pylimid import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": Posterior(values=Array([...]), states=("no", "yes"))}

    solution = solve(diagram)
    # → Solution(policy={"treat": {(0,): 0, (1,): 1}}, expected_utility=78.3,
    #            method="backward_induction")
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import jax

from pylimid.graph.diagram import DiagramSnapshot, InfluenceDiagram
from pylimid.graph.node import NodeKind
from pylimid.inference._errors import InferenceError
from pylimid.inference.numpyro import samples as numpyro_samples
from pylimid.inference.numpyro.solvers import (
    backward_induction_solve,
    batched_solve,
    is_solvable,
)
from pylimid.inference.posterior import InferenceResult, Posterior
from pylimid.inference.solution import Policy, Solution, SolverName

__all__ = [
    "InferenceError",
    "InferenceResult",
    "Policy",
    "Posterior",
    "Solution",
    "SolveMethod",
    "SolverName",
    "infer",
    "solve",
]

#: Solver selection for solve().
SolveMethod = Literal["auto", "backward_induction", "scan"]


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int | float] | None = None,
    policy: Mapping[str, int | dict[tuple[int, ...], int]] | None = None,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Only the query and observed nodes and their ancestors take part in
    inference: any other node cannot change the posterior, so it is left out.
    A decision among those ancestors must be bound by *policy*. A fixed
    integer action clamps it, while a table policy selects an action from its
    information-set assignment. A decision outside them needs no policy.

    A utility in *query* is computed from its parents in every draw, which
    gives the distribution of the payoff under the bound actions and the
    evidence. Utilities never affect the posterior of other variables.

    With no observed chance node, draws come from forward sampling. With evidence,
    NumPyro uses exact enumeration when every unobserved chance variable is discrete,
    and NUTS when some are continuous. Returned draws and their summaries are Monte
    Carlo estimates in all cases.

    The whole diagram is validated before selecting the relevant nodes. Chance
    distributions and utility functions must be JAX-traceable.

    Args:
        diagram: The influence diagram to query. Validation runs automatically.
        query: Names of chance and utility nodes whose posteriors are requested.
            Chance nodes may have scalar or array-valued draws; utilities are scalar.
        observed: Map from chance node name to its observed value: an integer state
            index for a discrete variable, a real value for a continuous one.
        policy: Map from decision names to fixed integer actions or contingent
            action tables in the format returned by `solve`. A table maps each
            tuple of parent-state indices, in the decision's ``parents`` order, to an
            action index in its ``states``. A decision with no parents uses ``()`` as
            its table key. Every decision that is an ancestor of a query or observed
            node must appear.
        num_samples: Number of posterior draws per query variable. NUTS
            runs 500 warmup steps before them.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        One `Posterior` per query variable, keyed by its name. Entries at the same
        position in each posterior's ``values`` belong to the same joint draw. Use
        `Posterior.marginal` for discrete probabilities, and `Posterior.mean`,
        `Posterior.std` or `Posterior.hdi` for continuous variables and utilities.
        Draws and numerical summaries are JAX arrays. Summaries have the node's shape,
        including zero-dimensional arrays for scalar nodes. 

    Raises:
        InferenceError: If a decision that influences the query or the
            evidence is unbound, if *policy* names a node that is not a
            decision or contains an incomplete or invalid action table, if
            *query* names a decision, or if *observed* names a node that is
            not a chance node.
        KeyError: If *query* or *observed* names a node that is not in the
            diagram.
        ValueError: If the diagram fails `InfluenceDiagram.validate`, or a table policy
            has a continuous information-set parent.
    """
    observed = observed or {}
    policy = policy or {}
    snapshot = diagram.snapshot()

    decisions = [
        name for name, node in snapshot.nodes if node.kind is NodeKind.DECISION
    ]
    unknown = sorted(set(policy) - set(decisions))
    if unknown:
        raise InferenceError(
            f"policy= binds decision nodes only; unknown names: {unknown}."
        )

    _validate_query(snapshot, query)
    _validate_observed(snapshot, observed)

    relevant = _ancestors(snapshot, [*query, *observed])
    unbound = [name for name in decisions if name in relevant and name not in policy]
    if unbound:
        raise InferenceError(
            f"Diagram has unbound decision nodes: {unbound}. They influence "
            "the query or the observed nodes, so specify policy={...} for them."
        )

    pruned = DiagramSnapshot(
        nodes=tuple((name, node) for name, node in snapshot.nodes if name in relevant)
    )
    table_policy: Policy = {}
    bound: dict[str, int] = {}
    for name, rule in policy.items():
        if name not in relevant:
            continue
        if isinstance(rule, dict):
            table_policy[name] = rule
        else:
            bound[name] = rule
    return _infer_numpyro(
        pruned,
        query,
        {**observed, **bound},
        num_samples=num_samples,
        rng_key=rng_key,
        policy=table_policy,
    )


def solve(
    diagram: InfluenceDiagram,
    *,
    method: SolveMethod = "auto",
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Find a policy that maximises expected total utility.

    Two Monte Carlo solvers are available, both for discrete decisions with discrete
    information sets. Chance nodes may be discrete, continuous, or both in the same
    diagram. The total utility is the sum of all utility nodes.

    - ``"backward_induction"`` resolves one decision at a time, from the
      last to the first, estimating each action's expected utility per
      information-set assignment by forward sampling. Its cost is additive
      in the decisions. It requires a solvable diagram (see
      `pylimid.inference.numpyro.solvers.is_solvable`), on which it
      finds the optimal policy up to sampling noise.
    - ``"scan"`` enumerates every policy and estimates each one's expected
      utility by forward sampling, keeping the best. It handles any
      diagram with discrete decisions and information sets, but the number of policies
      grows exponentially with the number of action-table entries.
    - ``"auto"`` uses backward induction when the diagram is solvable and
      the scan otherwise.

    Chance distributions and utility functions must be JAX-traceable: use ``jnp``
    operations on parent values, without ``float()`` or ``int()`` conversions or Python
    conditionals on those values.

    Args:
        diagram: An influence diagram with at least one decision. Validation runs
            automatically.
        method: Which solver to run.
        num_samples: Forward samples per evaluation: per candidate policy
            for the scan, per action per decision for backward induction.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        A `Solution` with the optimal per-decision policy (decision name to
        information-set assignment to chosen action), the expected total
        utility (a Monte-Carlo estimate), and the solver that ran
        (`Solution.method`, never ``"auto"``).

    Raises:
        InferenceError: If the diagram has no decision nodes, if a
            decision's information set is not discrete, if
            ``method="backward_induction"`` is asked for on a diagram that is
            not solvable, or if an information-set assignment is never
            sampled during backward induction. Increasing ``num_samples`` can help with
            rare assignments; impossible assignments cannot be sampled.
        ValueError: If the diagram fails `InfluenceDiagram.validate`, or *method* is not
            one of the supported names.
    """
    if method not in ("auto", "backward_induction", "scan"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'auto', "
            "'backward_induction' or 'scan'."
        )

    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    if method == "auto":
        method = "backward_induction" if is_solvable(snapshot) else "scan"
    solver = (
        backward_induction_solve if method == "backward_induction" else batched_solve
    )

    try:
        return solver(snapshot, num_samples=num_samples, rng_key=rng_key)
    except ValueError as e:
        raise InferenceError(f"solve(method={method!r}) failed. {e}") from e


def _validate_query(snapshot, query: list[str]) -> None:
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is NodeKind.DECISION:
            raise InferenceError(
                f"query= names chance and utility nodes only; {name!r} is a "
                "decision node. Decisions are clamped by policy=, not inferred."
            )


def _validate_observed(snapshot, observed: dict[str, int | float]) -> None:
    for name in observed:
        node = _find_node(snapshot, name)
        if node.kind is not NodeKind.CHANCE:
            raise InferenceError(
                f"observed= names chance nodes only; {name!r} is a "
                f"{node.kind.value} node. Bind decisions with policy=; "
                "utilities are computed, not observed."
            )


# -- backend ------------------------------------------------------------------


def _ancestors(snapshot: DiagramSnapshot, names: list[str]) -> set[str]:
    """*names* together with every node upstream of them."""
    parents = {name: node.parents for name, node in snapshot.nodes}
    found: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(parents[name])
    return found


def _infer_numpyro(
    snapshot,
    query,
    observed,
    *,
    num_samples: int,
    rng_key: jax.Array | None,
    policy: Policy,
) -> InferenceResult:
    # ``samples()`` returns raw draws; ``infer()`` wraps them in a
    # Posterior, keeping the draws and the variable's states together.
    # Everything is a Monte-Carlo estimate.
    draws = numpyro_samples(
        snapshot,
        observed=observed,
        query=query,
        num_samples=num_samples,
        rng_key=rng_key,
        policy=policy,
    )
    result: InferenceResult = {}
    for name in query:
        node = _find_node(snapshot, name)
        states = node.states if node.kind is NodeKind.CHANCE else None
        result[name] = Posterior(values=draws[name], states=states)
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
