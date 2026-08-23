"""
Unified inference entry-point.

::

    from decisionpy.inference import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": Marginal(values=[0.456, 0.544], exact=True)}

    solution = solve(diagram)
    # → Solution(policy={"treat": {(0,): 0, (1,): 1}}, expected_utility=78.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp

from decisionpy.graph.diagram import InfluenceDiagram
from decisionpy.graph.node import NodeKind
from decisionpy.inference.id.bucket_elim import (
    Policy,
    Solution,
)
from decisionpy.inference.id.bucket_elim import (
    solve as bucket_elim_solve,
)
from decisionpy.inference.numpyro import samples as numpyro_samples
from decisionpy.inference.ve import query as ve_query

__all__ = [
    "Draws",
    "InferenceError",
    "InferenceResult",
    "Marginal",
    "Policy",
    "Solution",
    "infer",
    "solve",
]


class InferenceError(Exception):
    """Asked for an inference method that cannot handle the given diagram."""


@dataclass(frozen=True)
class Marginal:
    """
    Posterior over a discrete variable: a probability vector over its states.

    Attributes:
        values: ``[P(0), P(1), ...]`` — sums to 1.
        exact: Whether ``values`` is exact (variable elimination) or a
            Monte-Carlo estimate (numpyro enumeration/MCMC draws).
    """

    values: list[float]
    exact: bool


@dataclass(frozen=True)
class Draws:
    """
    Posterior over a continuous variable: raw posterior draws.

    A continuous variable has no per-state probability mass, so its result
    is the raw sample list.

    Attributes:
        values: One posterior draw per entry.
    """

    values: list[float]


#: Result of :func:`infer`: one entry per query variable. A discrete variable
#: (declared ``states``) maps to a :class:`Marginal`; a continuous variable
#: maps to a :class:`Draws`.
InferenceResult: TypeAlias = dict[str, Marginal | Draws]


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int] | None = None,
    policy: dict[str, int] | None = None,
    engine: str = "auto",
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Bayesian networks infer directly. An influence diagram can be inferred
    once every decision is bound by *policy*: each bound decision is clamped
    to its chosen action and behaves like observed evidence, utilities are
    ignored, and the diagram collapses to a Bayesian network. Binding is
    all-or-nothing — partially bound is still ambiguous.

    Args:
        diagram: A validated influence diagram.
        query: Names of chance variables whose posteriors are requested.
        observed: Map from node name to its observed integer state.
        policy: Map from decision name to its chosen integer action. Every
            decision in *diagram* must appear.
        engine: ``"auto"`` (default), ``"ve"``, or ``"numpyro"``.

    Returns:
        ``{var_name: result}`` — one entry per query variable. A discrete
        variable (declared ``states``) maps to a :class:`Marginal` (its
        probability vector, plus whether it is exact); a continuous variable
        maps to a :class:`Draws` (raw posterior draws).

    Raises:
        InferenceError: If a decision is unbound, if *policy* names a node
            that is not a decision, if *query* names a non-chance node, or
            if the chosen engine cannot handle the diagram.
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

    evidence = {**observed, **policy}

    if engine == "auto":
        engine = _choose_engine(snapshot)

    if engine == "ve":
        return _infer_ve(snapshot, query, evidence)

    if engine == "numpyro":
        return _infer_numpyro(snapshot, query, evidence)

    raise InferenceError(
        f"Unknown engine {engine!r}. Choose 'auto', 've', or 'numpyro'."
    )


def solve(diagram: InfluenceDiagram, *, engine: str = "auto") -> Solution:
    """
    Solve an influence diagram: the optimal policy and its expected utility.

    Args:
        diagram: A validated influence diagram with at least one decision.
        engine: ``"auto"`` (default) or ``"bucket_elim"``.

    Returns:
        A :class:`Solution` with the optimal per-decision policy (decision
        name to information-set assignment to chosen action) and the
        maximum expected total utility.

    Raises:
        InferenceError: If the diagram has no decision nodes, or if the
            chosen engine cannot handle the diagram.
    """
    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    if engine == "auto":
        engine = _choose_solver(snapshot)

    if engine == "bucket_elim":
        return _solve_bucket_elim(snapshot)

    if engine == "numpyro":
        raise InferenceError(
            "The NumPyro intervention-scan solver for mixed/continuous "
            "influence diagrams is not implemented yet; bucket elimination "
            "handles all-categorical diagrams."
        )

    raise InferenceError(f"Unknown engine {engine!r}. Choose 'auto' or 'bucket_elim'.")


# -- engine selection ---------------------------------------------------------


def _choose_engine(snapshot) -> str:
    # Discrete chance nodes decide between the exact (ve) and Monte-Carlo
    # (numpyro) engines. Decisions are clamped constants in either engine,
    # and a utility node's ``is_discrete`` is vacuous (always ``False``) —
    # only chance nodes are ever classified.
    if all(
        node.is_discrete for _, node in snapshot.nodes if node.kind is NodeKind.CHANCE
    ):
        return "ve"
    return "numpyro"


def _choose_solver(snapshot) -> str:
    # Decisions are categorical by construction (``states`` are mandatory),
    # so continuous chance nodes are the only thing that rules bucket
    # elimination out — same kind-based classification as ``infer()``.
    if all(
        node.is_discrete for _, node in snapshot.nodes if node.kind is NodeKind.CHANCE
    ):
        return "bucket_elim"
    return "numpyro"


def _validate_query(snapshot, query: list[str]) -> None:
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is not NodeKind.CHANCE:
            raise InferenceError(
                f"query= names chance variables only; {name!r} is a "
                f"{node.kind.value} node. Decisions are clamped by policy= "
                f"and utilities are not random variables."
            )


# -- engine backends ----------------------------------------------------------


def _infer_ve(snapshot, query, observed) -> InferenceResult:
    try:
        vectors = ve_query(snapshot, variables=query, observed=observed)
    except TypeError as e:
        raise InferenceError(
            f"Variable elimination requires an all-discrete diagram. {e}"
        ) from e
    return {
        name: Marginal(values=values, exact=True) for name, values in vectors.items()
    }


def _infer_numpyro(snapshot, query, observed) -> InferenceResult:
    # ``samples()`` returns raw draws; ``infer()`` normalizes them to the
    # per-type contract: a Marginal for discrete variables, raw Draws for
    # continuous ones. NumPyro results are Monte-Carlo estimates (enumerated
    # draws bincounted), never exact.
    draws = numpyro_samples(snapshot, observed=observed, query=query)
    result: InferenceResult = {}
    for name in query:
        node = _find_node(snapshot, name)
        if node.is_discrete:
            counts = jnp.bincount(draws[name], length=len(node.states))
            probs = counts / counts.sum()
            result[name] = Marginal(values=[float(p) for p in probs], exact=False)
        else:
            result[name] = Draws(values=[float(x) for x in draws[name]])
    return result


def _solve_bucket_elim(snapshot) -> Solution:
    try:
        return bucket_elim_solve(snapshot)
    except TypeError as e:
        raise InferenceError(
            f"Bucket elimination requires an all-categorical diagram. {e}"
        ) from e


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
