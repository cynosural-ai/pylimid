"""
GaussianFactor — a Gaussian function over named continuous variables.

Internal to the linear-Gaussian inference engine. A factor represents

    phi(x) = exp(-1/2 x^T J x + h^T x)

over its variables — the canonical (information) form of a Gaussian,
where J is the precision matrix and h = J mu. Products and marginals of
Gaussians stay Gaussian, so exact inference reduces to three operations:

- multiply (two factors that share variables): add J and h, aligned to
  the union of scopes;
- marginal (sum variables out): the Schur complement;
- condition (clamp variables to observed values): delete the variable and
  fold its value into h.

This is the continuous analog of utils.factor.Factor: variable
elimination over these factors yields exact Gaussian posteriors. Constants
are dropped throughout — a marginal factor is determined only up to
scale, which is all the elimination needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

__all__ = ["GaussianFactor"]


class GaussianFactor:
    """
    A Gaussian function over a set of continuous variables.

    Attributes:
        variables: Names of the variables, in order.
        precision: The precision matrix J (symmetric, n x n).
        info: The information vector h = J mu (n,).
    """

    def __init__(
        self,
        *,
        variables: Sequence[str],
        precision: jnp.ndarray,
        info: jnp.ndarray,
    ) -> None:
        """Create a canonical-form factor over *variables*."""
        if len(set(variables)) != len(variables):
            raise ValueError(f"Duplicate variable names: {variables}.")
        if precision.shape != (len(variables), len(variables)):
            raise ValueError(
                f"precision must be ({len(variables)}, {len(variables)}), "
                f"got {precision.shape}."
            )
        if info.shape != (len(variables),):
            raise ValueError(f"info must be ({len(variables)},), got {info.shape}.")
        self.variables = tuple(variables)
        self.precision = jnp.asarray(precision, dtype=float)
        self.info = jnp.asarray(info, dtype=float)

    # -- representation -------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact representation."""
        return f"GaussianFactor({self.variables})"

    # -- core operations ------------------------------------------------------

    def __mul__(self, other: GaussianFactor) -> GaussianFactor:
        """
        Product of two factors, aligned to the union of their scopes.

        The union keeps *self*'s variable order and appends *other*'s
        remaining variables in order. Shared variables add their J and h
        entries.
        """
        variables = list(self.variables)
        variables.extend(v for v in other.variables if v not in self.variables)
        n = len(variables)
        precision = jnp.zeros((n, n))
        info = jnp.zeros((n,))
        for factor in (self, other):
            rows = jnp.asarray([variables.index(v) for v in factor.variables])
            precision = precision.at[rows[:, None], rows[None, :]].add(
                factor.precision
            )
            info = info.at[rows].add(factor.info)
        return GaussianFactor(variables=variables, precision=precision, info=info)

    def marginal(self, variables: Sequence[str]) -> GaussianFactor:
        """
        Marginalize out every variable not in *variables* (Schur complement).

        Args:
            variables: Names to keep; the rest are summed out.

        Returns:
            A factor over exactly those variables, ordered as in *self*.
        """
        keep = sorted(
            (v for v in variables if v in self.variables),
            key=lambda v: self.variables.index(v),
        )
        if keep == list(self.variables):
            return self
        keep_idx = jnp.asarray([self.variables.index(v) for v in keep])
        remove_idx = jnp.asarray(
            [i for i, v in enumerate(self.variables) if v not in keep]
        )
        j_kk = self.precision[jnp.ix_(keep_idx, keep_idx)]
        j_kr = self.precision[jnp.ix_(keep_idx, remove_idx)]
        j_rr = self.precision[jnp.ix_(remove_idx, remove_idx)]
        h_k = self.info[keep_idx]
        h_r = self.info[remove_idx]
        coupling = j_kr @ jnp.linalg.inv(j_rr)
        return GaussianFactor(
            variables=keep,
            precision=j_kk - coupling @ j_kr.T,
            info=h_k - coupling @ h_r,
        )

    def condition(self, evidence: dict[str, float]) -> GaussianFactor:
        """
        Clamp the given variables to their observed values.

        Args:
            evidence: Variable name to observed value.

        Returns:
            A factor over the remaining variables.
        """
        kept = [
            i for i, v in enumerate(self.variables) if v not in evidence
        ]
        if len(kept) == len(self.variables):
            return self
        removed = [
            (i, v) for i, v in enumerate(self.variables) if v in evidence
        ]
        kept_idx = jnp.asarray(kept)
        remove_idx = jnp.asarray([i for i, _ in removed])
        values = jnp.asarray([evidence[v] for _, v in removed])
        return GaussianFactor(
            variables=[v for v in self.variables if v not in evidence],
            precision=self.precision[jnp.ix_(kept_idx, kept_idx)],
            info=self.info[kept_idx]
            - self.precision[jnp.ix_(kept_idx, remove_idx)] @ values,
        )

    @classmethod
    def from_cpd(
        cls,
        variable: str,
        parents: Sequence[str],
        intercept: float,
        slopes: Sequence[float],
        scale: float,
    ) -> GaussianFactor:
        """
        Build the factor for x | parents ~ Normal(a + b^T p, scale^2).

        Args:
            variable: The child variable name.
            parents: Its parent names, in order.
            intercept: The conditional-mean intercept a.
            slopes: The linear coefficients b, one per parent.
            scale: The conditional standard deviation.
        """
        b = jnp.asarray(slopes, dtype=float)
        parents = list(parents)
        n_parents = len(parents)
        n = n_parents + 1
        variance = scale**2
        precision = jnp.zeros((n, n))
        precision = precision.at[:n_parents, :n_parents].set(
            jnp.outer(b, b) / variance
        )
        precision = precision.at[:n_parents, n_parents].set(-b / variance)
        precision = precision.at[n_parents, :n_parents].set(-b / variance)
        precision = precision.at[n_parents, n_parents].set(1.0 / variance)
        info = jnp.zeros((n,))
        info = info.at[:n_parents].set(-intercept * b / variance)
        info = info.at[n_parents].set(intercept / variance)
        return cls(
            variables=[*parents, variable],
            precision=precision,
            info=info,
        )
