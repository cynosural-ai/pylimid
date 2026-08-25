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
from decisionpy.inference.exact.categorical import (
    Policy,
    Solution,
)
from decisionpy.inference.exact.categorical import (
    query as ve_query,
)
from decisionpy.inference.exact.categorical import (
    solve as bucket_elim_solve,
)
from decisionpy.inference.exact.linear_gaussian import query as lg_query
from decisionpy.inference.numpyro import samples as numpyro_samples

__all__ = [
    "Draws",
    "Gaussian",
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


@dataclass(frozen=True)
class Gaussian:
    """
    Exact posterior over a continuous variable of a linear-Gaussian BN.

    Attributes:
        mean: The posterior mean.
        variance: The posterior variance.
    """

    mean: float
    variance: float


#: Result of infer(): one entry per query variable. A discrete variable
#: (declared ``states``) maps to a Marginal; a continuous variable maps to
#: a Draws (numpyro) or a Gaussian (exact linear-Gaussian engine).
InferenceResult: TypeAlias = dict[str, Marginal | Draws | Gaussian]


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int | float] | None = None,
    policy: dict[str, int] | None = None,
    engine: str = "numpyro",
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
        observed: Map from node name to its observed value — an integer
            state for a discrete variable, a real value for a continuous
            one.
        policy: Map from decision name to its chosen integer action. Every
            decision in *diagram* must appear.
        engine: ``"numpyro"`` (default, Monte-Carlo), ``"ve"`` (exact
            discrete), or ``"lg"`` (exact linear-Gaussian). The exact
            engines are explicit opt-ins: they validate the diagram
            against their preconditions and fail loudly when it does not
            satisfy them.

    Returns:
        ``{var_name: result}`` — one entry per query variable. A discrete
        variable (declared ``states``) maps to a Marginal (its
        probability vector, plus whether it is exact); a continuous variable
        maps to a Draws (raw posterior draws) under numpyro, or a Gaussian
        (exact mean and variance) under the linear-Gaussian engine.

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

    if engine == "ve":
        return _infer_ve(snapshot, query, evidence)

    if engine == "lg":
        return _infer_lg(snapshot, query, evidence)

    if engine == "numpyro":
        return _infer_numpyro(snapshot, query, evidence)

    raise InferenceError(f"Unknown engine {engine!r}. Choose 'numpyro', 've', or 'lg'.")


def solve(diagram: InfluenceDiagram, *, engine: str = "auto") -> Solution:
    """
    Solve an influence diagram: the optimal policy and its expected utility.

    Args:
        diagram: A validated influence diagram with at least one decision.
        engine: ``"auto"`` (default) or ``"bucket_elim"``.

    Returns:
        A Solution with the optimal per-decision policy (decision
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


# -- solver selection ---------------------------------------------------------


def _choose_solver(snapshot) -> str:
    # Decisions are categorical by construction (``states`` are mandatory),
    # so continuous chance nodes are the only thing that rules bucket
    # elimination out.
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


def _infer_lg(snapshot, query, observed) -> InferenceResult:
    # A CPD that is not a linear Gaussian is rejected loudly by the probe
    # (TypeError for a non-Normal, ValueError for a non-affine loc) and
    # surfaced here as an InferenceError.
    try:
        posteriors = lg_query(snapshot, variables=query, observed=observed)
    except (TypeError, ValueError) as e:
        raise InferenceError(
            f"The linear-Gaussian engine requires an all-continuous diagram "
            f"of Normal CPDs. {e}"
        ) from e
    return {
        name: Gaussian(mean=mean, variance=variance)
        for name, (mean, variance) in posteriors.items()
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
