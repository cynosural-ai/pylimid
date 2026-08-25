"""
Typed results shared by the inference engines.

Engine modules produce these; the engine entry-points (infer, solve)
normalize to them. Kept engine-agnostic so no engine imports another
engine's internals: the numpyro solver and bucket elimination both
construct a Solution without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "Draws",
    "Gaussian",
    "InferenceResult",
    "Marginal",
    "Policy",
    "Solution",
]


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


#: Optimal policy: decision name -> {information-set assignment: chosen action}.
Policy: TypeAlias = dict[str, dict[tuple[int, ...], int]]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: Per-decision optimal action for every instantiation of the
            decision's information set.
        expected_utility: Expected total utility under *policy* — the exact
            maximum (bucket elimination) or a Monte-Carlo estimate
            (numpyro intervention scan), per *exact*.
        exact: Whether *expected_utility* is exact or a Monte-Carlo estimate.
    """

    policy: Policy
    expected_utility: float
    exact: bool
