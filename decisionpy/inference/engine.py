"""
Unified inference entry-point.

::

    from decisionpy.inference import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": Posterior(values=[...], states=("no", "yes"))}

    solution = solve(diagram)
    # → Solution(policy={"treat": {(0,): 0, (1,): 1}}, expected_utility=78.3)
"""

from __future__ import annotations

from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import NodeKind
from decisionpy.inference.numpyro import samples as numpyro_samples
from decisionpy.inference.numpyro.solver import solve as numpyro_solve
from decisionpy.inference.result import (
    InferenceResult,
    Policy,
    Posterior,
    Solution,
)

__all__ = [
    "InferenceError",
    "InferenceResult",
    "Policy",
    "Posterior",
    "Solution",
    "infer",
    "solve",
]


class InferenceError(Exception):
    """Asked for an inference method that cannot handle the given diagram."""


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int | float] | None = None,
    policy: dict[str, int] | None = None,
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Bayesian networks infer directly. An influence diagram can be inferred
    once every decision is bound by *policy*: each bound decision is clamped
    to its chosen action and behaves like observed evidence, utilities are
    ignored, and the diagram collapses to a Bayesian network. Binding is
    all-or-nothing — partially bound is still ambiguous.

    The NumPyro engine is the only engine: posterior draws via exact
    discrete enumeration for all-discrete diagrams and NUTS otherwise, so
    every result is a Monte-Carlo estimate.

    Args:
        diagram: A validated influence diagram.
        query: Names of chance variables whose posteriors are requested.
        observed: Map from node name to its observed value — an integer
            state for a discrete variable, a real value for a continuous
            one.
        policy: Map from decision name to its chosen integer action. Every
            decision in *diagram* must appear.

    Returns:
        ``{var_name: result}`` — one Posterior per query variable: raw
        posterior draws plus the variable's states (None for continuous).
        Each summary method is valid for exactly one kind: marginal()
        bincounts a discrete posterior into its probability vector, and
        mean()/std()/hdi() summarize the draws of a continuous one —
        calling the wrong kind raises. Everything is a Monte-Carlo
        estimate.

    Raises:
        InferenceError: If a decision is unbound, if *policy* names a node
            that is not a decision, or if *query* names a non-chance node.
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
    unbound = [name for name in decisions if name not in policy]
    if unbound:
        message = (
            f"Diagram has unbound decision nodes: {unbound}. "
            "Specify policy={...} or call solve() first."
        )
        bound = [name for name in decisions if name in policy]
        if bound:
            message += (
                " Currently bound: "
                f"{', '.join(f'{name}={policy[name]}' for name in bound)}."
            )
        raise InferenceError(message)

    _validate_query(snapshot, query)

    return _infer_numpyro(snapshot, query, {**observed, **policy})


def solve(diagram: InfluenceDiagram) -> Solution:
    """
    Solve an influence diagram: the optimal policy and its expected utility.

    Runs the NumPyro intervention scan (Strategy B): enumerates the
    discrete policy space and estimates expected utility per policy by
    forward sampling with each decision resolved from its observed
    information set.

    Args:
        diagram: A validated influence diagram with at least one decision.

    Returns:
        A Solution with the optimal per-decision policy (decision
        name to information-set assignment to chosen action) and the
        expected total utility, a Monte-Carlo estimate.

    Raises:
        InferenceError: If the diagram has no decision nodes, or if a
            decision's information set is not discrete.
    """
    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    try:
        return numpyro_solve(snapshot)
    except ValueError as e:
        raise InferenceError(
            f"The intervention-scan solver cannot handle this diagram. {e}"
        ) from e


def _validate_query(snapshot, query: list[str]) -> None:
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is not NodeKind.CHANCE:
            raise InferenceError(
                f"query= names chance variables only; {name!r} is a "
                f"{node.kind.value} node. Decisions are clamped by policy= "
                f"and utilities are not random variables."
            )


# -- backend ------------------------------------------------------------------


def _infer_numpyro(snapshot, query, observed) -> InferenceResult:
    # ``samples()`` returns raw draws; ``infer()`` wraps them in a
    # Posterior, keeping the draws and the variable's states together.
    # Everything is a Monte-Carlo estimate.
    draws = numpyro_samples(snapshot, observed=observed, query=query)
    result: InferenceResult = {}
    for name in query:
        node = _find_node(snapshot, name)
        result[name] = Posterior(
            values=[float(x) for x in draws[name]], states=node.states
        )
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
