"""
Posterior draws for a query variable — the result of `infer`.

Returned by `infer`. Draws and summaries are Monte Carlo estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TypeAlias

import jax
import jax.numpy as jnp

__all__ = ["InferenceResult", "Posterior"]


@dataclass(frozen=True)
class Posterior:
    """
    Posterior draws and state labels for one query variable.

    Returned by `infer`. Draws retain the sampler's dtype: categorical variables contain
    integer state indices, and numerical variables and utilities contain numerical
    values. All numerical results are JAX arrays and all summaries are Monte Carlo
    estimates.

    Use `marginal` for category probabilities, and `mean`, `std` or `hdi` for numerical
    variables and utilities. Each method raises ``ValueError`` when used with the other
    kind. For array-valued variables, summaries are computed separately for each
    component and retain the variable's shape.

    Attributes:
        values: JAX array with shape ``(num_samples, *node_shape)``. Scalar nodes have
            shape ``(num_samples,)``. Posteriors returned by the same `infer` call share
            the same draw order.
        states: State labels for a categorical variable; ``None`` for numerical
            variables and utilities.
    """

    values: jax.Array
    states: tuple[str, ...] | None = None

    @property
    def is_categorical(self) -> bool:
        """Whether the draws represent named categories rather than numerical values."""
        return self.states is not None

    def marginal(self) -> jax.Array:
        """
        Estimate the probability of each named category from the draws.

        Returns:
            JAX array of probabilities in ``states`` order, summing to 1.

        Raises:
            ValueError: If the variable has no named categories (``states`` is
                ``None``), or a draw is outside the declared state indices.
        """
        states = self._require_categorical()
        if bool(jnp.any((self.values < 0) | (self.values >= len(states)))):
            raise ValueError(
                f"Categorical posterior draws must be state indices from 0 to "
                f"{len(states) - 1}."
            )
        counts = jnp.bincount(self.values, length=len(states))
        return counts / self.values.shape[0]

    def mean(self) -> jax.Array:
        """
        The mean of the posterior draws.

        Returns:
            JAX array with the node's shape containing the estimated mean. Scalar
            nodes return a zero-dimensional array.

        Raises:
            ValueError: If the variable is categorical (``states`` set): the
                mean of state indices depends on their encoding; use
                `marginal` for the probability vector instead.
        """
        self._require_numeric()
        return jnp.mean(self.values, axis=0)

    def std(self) -> jax.Array:
        """
        The population standard deviation of the posterior draws.

        Returns:
            JAX array with the node's shape containing the estimated spread. Scalar
            nodes return a zero-dimensional array; use ``.item()`` for a Python scalar.

        Raises:
            ValueError: If the variable is categorical (``states`` set).
        """
        self._require_numeric()
        return jnp.std(self.values, axis=0)

    def hdi(self, prob: float = 0.95) -> tuple[jax.Array, jax.Array]:
        """
        The highest-density interval covering ``prob`` of the draws.

        The narrowest interval spanning ``ceil(prob * n)`` sorted draws. For an
        array-valued variable, each component has its own interval; the intervals
        do not define a joint credible region.

        Args:
            prob: Coverage in (0, 1]. Defaults to 0.95.

        Returns:
            ``(lower, upper)`` bounds as two JAX arrays with the node's shape. Scalar
            nodes return zero-dimensional arrays.

        Raises:
            ValueError: If *prob* is not in (0, 1], or if the variable is
                categorical (``states`` set).
        """
        if not 0 < prob <= 1:
            raise ValueError(f"hdi() coverage must be in (0, 1], got {prob}.")
        self._require_numeric()
        k = ceil(prob * len(self.values))
        ordered_array = jnp.sort(self.values, axis=0)
        widths = ordered_array[k - 1 :] - ordered_array[: len(self.values) - k + 1]
        starts = jnp.argmin(widths, axis=0, keepdims=True)
        lower = jnp.take_along_axis(ordered_array, starts, axis=0)[0]
        upper = jnp.take_along_axis(ordered_array, starts + k - 1, axis=0)[0]
        return (lower, upper)

    def _require_categorical(self) -> tuple[str, ...]:
        """
        Fail loudly on a numerical posterior, else return the states.

        The return value is the narrowing: the caller gets the guaranteed
        non-None states from the type checker's perspective.
        """
        if self.states is None:
            raise ValueError(
                "marginal() requires a categorical variable; this posterior is "
                "numerical (states is None). Summarize with mean(), std(), "
                "hdi(), or use the raw values."
            )
        return self.states

    def _require_numeric(self) -> None:
        """Fail loudly on a categorical posterior: indices are not moments."""
        if self.states is not None:
            raise ValueError(
                "this summary requires a numerical variable; the posterior "
                f"is categorical (states = {self.states}). Use marginal() for "
                "the probability vector, or compute over the raw index "
                "draws explicitly if that is what you mean."
            )


#: Result of infer(): one Posterior per query variable. Every variable maps
#: to the same type — raw draws plus its states (None when numerical);
#: marginal() recovers the probability vector for categorical variables.
#: Everything is a Monte-Carlo estimate.
InferenceResult: TypeAlias = dict[str, Posterior]
