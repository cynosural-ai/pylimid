"""
Posterior draws for a query variable — the result of `infer`.

Engine modules produce these; the entry point normalizes to them. With a
single NumPyro engine, every posterior is a Monte-Carlo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import pstdev
from typing import TypeAlias

__all__ = ["InferenceResult", "Posterior"]


@dataclass(frozen=True)
class Posterior:
    """
    Posterior draws and state labels for one query variable.

    Returned by `infer`. For a discrete variable, the draws are state indices stored as
    floats; for a continuous variable or a utility, they are numerical values. All
    summaries are Monte Carlo estimates.

    Use `marginal` for discrete probabilities, and `mean`, `std` or `hdi` for continuous
    variables and utilities. Each method raises ``ValueError`` when used with the other
    kind.

    Attributes:
        values: One scalar posterior draw per entry. Posteriors returned by the same
            `infer` call share the same draw order.
        states: State labels for a discrete variable; ``None`` for a continuous variable
            or a utility.
    """

    values: list[float]
    states: tuple[str, ...] | None = None

    @property
    def is_discrete(self) -> bool:
        """Whether the variable has declared state labels."""
        return self.states is not None

    def marginal(self) -> list[float]:
        """
        Estimate the discrete probability of each state from the draws.

        Returns:
            Probabilities in ``states`` order, summing to 1.

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

        Returns:
            The estimated mean of the continuous variable or utility.

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

        Returns:
            The estimated spread of the continuous variable or utility.

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
