"""
Typed results shared by the inference engines.

Engine modules produce these; the engine entry-points (infer, solve)
normalize to them. With a single NumPyro engine, every result is a
Monte-Carlo estimate — the contract carries no exactness flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

__all__ = ["Draws", "InferenceResult", "Marginal", "Policy", "Solution"]


@dataclass(frozen=True)
class Marginal:
    """
    Posterior over a discrete variable: a probability vector over its states.

    A Monte-Carlo estimate: enumerated draws bincounted by ``infer()``.

    Attributes:
        values: ``[P(0), P(1), ...]`` — sums to 1.
    """

    values: list[float]


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


#: Result of infer(): one entry per query variable. A discrete variable
#: (declared ``states``) maps to a Marginal; a continuous variable maps to
#: raw Draws. Everything is a Monte-Carlo estimate.
InferenceResult: TypeAlias = dict[str, Marginal | Draws]


#: Optimal policy: decision name -> {information-set assignment: chosen action}.
Policy: TypeAlias = dict[str, dict[tuple[int, ...], int]]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: Per-decision optimal action for every instantiation of the
            decision's information set.
        expected_utility: Monte-Carlo estimate of the expected total
            utility under *policy* (numpyro intervention scan).
    """

    policy: Policy
    expected_utility: float
