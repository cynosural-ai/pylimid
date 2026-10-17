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

import jax
import jax.numpy as jnp

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
    kind. For array-valued variables, summaries are computed separately for each
    component and retain the variable's shape.

    Attributes:
        values: A list of floats for scalar draws, or a JAX array with shape
            ``(num_samples, *node_shape)`` for array-valued draws. Posteriors returned
            by the same `infer` call share the same draw order.
        states: State labels for a discrete variable; ``None`` for a continuous variable
            or a utility.
    """

    values: list[float] | jax.Array
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

    def mean(self) -> float | jax.Array:
        """
        The mean of the posterior draws.

        Returns:
            The estimated mean of the continuous variable or utility. Scalar draws
            return a float; array-valued draws return a JAX array with the node's shape.

        Raises:
            ValueError: If the variable is discrete (``states`` set): the
                mean of state indices depends on their encoding; use
                `marginal` for the probability vector instead.
        """
        self._require_continuous()
        if isinstance(self.values, list):
            return sum(self.values) / len(self.values)
        return jnp.mean(self.values, axis=0)

    def std(self) -> float | jax.Array:
        """
        The population standard deviation of the posterior draws.

        Returns:
            The estimated spread of the continuous variable or utility. Scalar draws
            return a float; array-valued draws return a JAX array with the node's shape.

        Raises:
            ValueError: If the variable is discrete (``states`` set).
        """
        self._require_continuous()
        if isinstance(self.values, list):
            return pstdev(self.values)
        return jnp.std(self.values, axis=0)

    def hdi(self, prob: float = 0.95) -> tuple[float | jax.Array, float | jax.Array]:
        """
        The highest-density interval covering ``prob`` of the draws.

        The narrowest interval spanning ``ceil(prob * n)`` sorted draws. For an
        array-valued variable, each component has its own interval; the intervals
        do not define a joint credible region.

        Args:
            prob: Coverage in (0, 1]. Defaults to 0.95.

        Returns:
            ``(lower, upper)`` bounds of the interval. Scalar draws return floats;
            array-valued draws return two JAX arrays with the node's shape.

        Raises:
            ValueError: If *prob* is not in (0, 1], or if the variable is
                discrete (``states`` set).
        """
        if not 0 < prob <= 1:
            raise ValueError(f"hdi() coverage must be in (0, 1], got {prob}.")
        self._require_continuous()
        k = ceil(prob * len(self.values))
        if isinstance(self.values, list):
            ordered = sorted(self.values)
            start = min(
                range(len(ordered) - k + 1),
                key=lambda i: ordered[i + k - 1] - ordered[i],
            )
            return (ordered[start], ordered[start + k - 1])
        ordered_array = jnp.sort(self.values, axis=0)
        widths = ordered_array[k - 1 :] - ordered_array[: len(self.values) - k + 1]
        starts = jnp.argmin(widths, axis=0, keepdims=True)
        lower = jnp.take_along_axis(ordered_array, starts, axis=0)[0]
        upper = jnp.take_along_axis(ordered_array, starts + k - 1, axis=0)[0]
        return (lower, upper)

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
