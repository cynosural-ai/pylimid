"""
Typed results shared by the inference engines.

Engine modules produce these; the entry points (`infer`, `solve`)
normalize to them. With a single NumPyro engine, every result is a
Monte-Carlo estimate — the contract carries no exactness flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import pstdev
from typing import Literal, TypeAlias

__all__ = ["InferenceResult", "Policy", "Posterior", "Solution", "SolverName"]


@dataclass(frozen=True)
class Posterior:
    """
    Posterior over a query variable: raw draws plus its state labels.

    A Monte-Carlo estimate: one posterior draw per entry, produced by the
    NumPyro engine (discrete enumeration or NUTS). Draws are the universal
    representation — a discrete variable's draws are state indices, a
    continuous variable's are real values. Summary methods derive from the
    draws, and every method is valid for exactly one variable kind: a
    discrete posterior's `marginal` bincounts the states into a
    probability vector, and a continuous posterior's `mean` / `std` /
    `hdi` summarize the draws. Each method raises on the other kind rather than
    silently computing something encoding-dependent.

    Attributes:
        values: One posterior draw per entry.
        states: State labels for a discrete variable; ``None`` for a
            continuous one.
    """

    values: list[float]
    states: tuple[str, ...] | None = None

    @property
    def is_discrete(self) -> bool:
        """Whether the variable has declared state labels."""
        return self.states is not None

    def marginal(self) -> list[float]:
        """
        The discrete probability vector over the states (bincounted draws).

        Returns:
            ``[P(0), P(1), ...]`` — sums to 1.

        Raises:
            ValueError: If the variable is continuous (``states`` is ``None``):
                a continuous posterior has no per-state probability mass.
        """
        states = self._require_discrete()
        counts = [0] * len(states)
        for value in self.values:
            counts[int(value)] += 1
        total = sum(counts)
        return [count / total for count in counts]

    def mean(self) -> float:
        """
        The mean of the posterior draws.

        Raises:
            ValueError: If the variable is discrete (``states`` set): the
                mean of state indices depends on their encoding; use
                `marginal` for the probability vector instead.
        """
        self._require_continuous()
        return sum(self.values) / len(self.values)

    def std(self) -> float:
        """
        The population standard deviation of the posterior draws.

        Raises:
            ValueError: If the variable is discrete (``states`` set).
        """
        self._require_continuous()
        return pstdev(self.values)

    def hdi(self, prob: float = 0.95) -> tuple[float, float]:
        """
        The highest-density interval covering ``prob`` of the draws.

        The narrowest interval spanning ``ceil(prob * n)`` sorted draws.

        Args:
            prob: Coverage in (0, 1]. Defaults to 0.95.

        Returns:
            ``(lower, upper)`` bounds of the interval.

        Raises:
            ValueError: If *prob* is not in (0, 1], or if the variable is
                discrete (``states`` set).
        """
        if not 0 < prob <= 1:
            raise ValueError(f"hdi() coverage must be in (0, 1], got {prob}.")
        self._require_continuous()
        ordered = sorted(self.values)
        k = ceil(prob * len(ordered))
        start = min(
            range(len(ordered) - k + 1),
            key=lambda i: ordered[i + k - 1] - ordered[i],
        )
        return (ordered[start], ordered[start + k - 1])

    def _require_discrete(self) -> tuple[str, ...]:
        """
        Fail loudly on a continuous posterior, else return the states.

        The return value is the narrowing: the caller gets the guaranteed
        non-None states from the type checker's perspective.
        """
        if self.states is None:
            raise ValueError(
                "marginal() requires a discrete variable; this posterior is "
                "continuous (states is None). Summarize with mean(), std(), "
                "hdi(), or use the raw values."
            )
        return self.states

    def _require_continuous(self) -> None:
        """Fail loudly on a discrete posterior: indices are not moments."""
        if self.states is not None:
            raise ValueError(
                "this summary requires a continuous variable; the posterior "
                f"is discrete (states = {self.states}). Use marginal() for "
                "the probability vector, or compute over the raw index "
                "draws explicitly if that is what you mean."
            )


#: Result of infer(): one Posterior per query variable. Every variable maps
#: to the same type — raw draws plus its states (None when continuous);
#: marginal() recovers the probability vector for discrete variables.
#: Everything is a Monte-Carlo estimate.
InferenceResult: TypeAlias = dict[str, Posterior]


#: Optimal policy: decision name -> {information-set assignment: chosen action}.
Policy: TypeAlias = dict[str, dict[tuple[int, ...], int]]

#: The solver that produced a `Solution`.
SolverName: TypeAlias = Literal["backward_induction", "scan"]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: Per-decision optimal action for every instantiation of the
            decision's information set.
        expected_utility: Monte-Carlo estimate of the expected total
            utility under *policy*.
        method: The solver that produced the solution. ``"scan"`` evaluated
            every policy, so *policy* is the best of the Monte-Carlo
            estimates; ``"backward_induction"`` resolved one decision at a
            time on a solvable diagram, optimal up to sampling noise. With
            ``solve(method="auto")`` this records which one was chosen.
    """

    policy: Policy
    expected_utility: float
    method: SolverName
