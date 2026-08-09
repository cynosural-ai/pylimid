r"""
Factor — a multidimensional array over named discrete variables.

Internal to the graph-native inference engines. A factor maps each assignment of
its variables to a non-negative number.

.. doctest::

    >>> f_rain = Factor(variables=["rain"], card=(2,), values=[0.8, 0.2])
    >>> f_wet = Factor(
    ...     variables=["rain", "wet_grass"],
    ...     card=(2, 2),
    ...     values=[[0.80, 0.20], [0.05, 0.95]],
    ... )
    >>> joint = f_rain * f_wet
    >>> joint.variables
    ['rain', 'wet_grass']
    >>> joint.marginal("wet_grass").values
    [0.35, 0.65]
"""

from __future__ import annotations

import itertools
from typing import Sequence


class Factor:
    """A non-negative function over a set of discrete variables."""

    __slots__ = ("variables", "card", "values")

    def __init__(
        self,
        *,
        variables: Sequence[str],
        card: Sequence[int],
        values: Sequence[float],
    ) -> None:
        """Create a factor over *variables* with given *card* and *values*."""
        self.variables = tuple(variables)
        self.card = tuple(card)
        flat = _flatten(values)
        if len(flat) != _prod(card):
            raise ValueError(
                f"Factor over {variables} with card {card} expects "
                f"{_prod(card)} values, got {len(flat)}."
            )
        self.values: list[float] = list(flat)

    # -- representation -------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact representation."""
        return f"Factor({self.variables}, card={self.card})"

    # -- core operations ------------------------------------------------------

    def __mul__(self, other: Factor) -> Factor:
        """
        Pointwise product, broadcasting over the union of variables.

        ::

            f(a,b) * g(b,c)  →  h(a,b,c) where h[a,b,c] = f[a,b] * g[b,c]
        """
        # Handle empty factors: an empty factor is the constant 1.
        if not self.variables:
            return Factor(
                variables=other.variables,
                card=other.card,
                values=list(other.values),
            )
        if not other.variables:
            return Factor(
                variables=self.variables,
                card=self.card,
                values=list(self.values),
            )

        all_vars = _union_vars(self.variables, other.variables)
        all_cards = {
            v: (
                self.card[self.variables.index(v)]
                if v in self.variables
                else other.card[other.variables.index(v)]
            )
            for v in all_vars
        }

        values: list[float] = []
        union_lookup = {v: i for i, v in enumerate(all_vars)}
        for assignment in _assignments(all_vars, all_cards):
            s = self[_project(assignment, self.variables, union_lookup)]
            o = other[_project(assignment, other.variables, union_lookup)]
            values.append(s * o)

        return Factor(
            variables=all_vars,
            card=tuple(all_cards[v] for v in all_vars),
            values=values,
        )

    def __add__(self, other: Factor) -> Factor:
        """
        Pointwise sum, broadcasting over the union of variables.

        Used to combine utility factors, whose objective is additive
        (``E[sum_k U_k]``) — the counterpart to :meth:`__mul__` for
        probabilities. Unlike :meth:`__mul__`, an empty-scope factor is a real
        constant here, not the unit 1, so it is broadcast like any other.

        ::

            f(a,b) + g(b,c)  →  h(a,b,c) where h[a,b,c] = f[a,b] + g[b,c]
        """
        all_vars = _union_vars(self.variables, other.variables)
        all_cards = {
            v: (
                self.card[self.variables.index(v)]
                if v in self.variables
                else other.card[other.variables.index(v)]
            )
            for v in all_vars
        }

        values: list[float] = []
        union_lookup = {v: i for i, v in enumerate(all_vars)}
        for assignment in _assignments(all_vars, all_cards):
            s = self[_project(assignment, self.variables, union_lookup)]
            o = other[_project(assignment, other.variables, union_lookup)]
            values.append(s + o)

        return Factor(
            variables=all_vars,
            card=tuple(all_cards[v] for v in all_vars),
            values=values,
        )

    def marginal(self, variables: Sequence[str]) -> Factor:
        """Sum out all variables not in *variables*."""
        keep = list(variables)
        drop = [v for v in self.variables if v not in keep]
        result = self
        # Eliminate in reverse order to keep axis indices stable during iteration.
        for v in reversed(drop):
            result = result._sum_out(v)
        # Reorder to match the requested variable order.
        return result._permute(keep)

    def condition(self, variable: str, value: int) -> Factor:
        """Fix *variable* to *value*, returning the reduced (one-smaller) factor."""
        idx = self.variables.index(variable)
        new_vars = list(self.variables)
        new_vars.remove(variable)

        # Build new values by slicing the dimension at *value*.
        new_card = list(self.card)
        del new_card[idx]
        n_before = _prod(self.card[:idx])
        stride = self.card[idx]
        n_after = _prod(self.card[idx + 1 :])

        new_values: list[float] = []
        for before in range(n_before):
            base = before * stride * n_after + value * n_after
            new_values.extend(self.values[base : base + n_after])

        return Factor(variables=new_vars, card=new_card, values=new_values)

    def normalize(self) -> Factor:
        """Divide all entries by their sum (in-place)."""
        total = sum(self.values)
        if total > 0:
            self.values = [v / total for v in self.values]
        elif not self.values:
            self.values = [1.0]  # empty factor → unit constant
        return self

    # -- internals ------------------------------------------------------------

    def _sum_out(self, variable: str) -> Factor:
        """Sum over *variable*, returning a factor with one fewer dimension."""
        idx = self.variables.index(variable)
        new_vars = list(self.variables)
        del new_vars[idx]
        new_card = list(self.card)
        del new_card[idx]

        if not new_vars:
            return Factor(variables=[], card=(), values=[sum(self.values)])

        n_before = _prod(self.card[:idx])
        stride = self.card[idx]
        n_after = _prod(self.card[idx + 1 :])

        new_values: list[float] = []
        for before in range(n_before):
            base = before * stride * n_after
            for after in range(n_after):
                total = 0.0
                for k in range(stride):
                    total += self.values[base + k * n_after + after]
                new_values.append(total)

        return Factor(variables=new_vars, card=new_card, values=new_values)

    def _permute(self, target_order: list[str]) -> Factor:
        """Reorder variables to match *target_order*."""
        if list(self.variables) == target_order:
            return self
        present = [v for v in target_order if v in self.variables]
        if not present:
            return self  # nothing to reorder, target vars were conditioned out
        target_order = present
        old_idx = [self.variables.index(v) for v in target_order]
        new_card = tuple(self.card[i] for i in old_idx)
        all_cards = {v: self.card_of(v) for v in self.variables}
        new_values: list[float] = []
        for assignment in _assignments(
            target_order, {v: all_cards[v] for v in target_order}
        ):
            old_lookup = {v: assignment[target_order.index(v)] for v in target_order}
            old_tuple = tuple(old_lookup.get(v, 0) for v in self.variables)
            new_values.append(self[old_tuple])
        return Factor(variables=target_order, card=new_card, values=new_values)

    # -- helpers --------------------------------------------------------------

    def card_of(self, variable: str) -> int:
        """Cardinality of *variable* (from another factor or precomputed)."""
        try:
            return self.card[self.variables.index(variable)]
        except ValueError as exc:
            raise KeyError(variable) from exc

    def __getitem__(self, key: tuple[int, ...]) -> float:
        """Flat-index lookup: ``factor[0, 1, 0]``."""
        assert len(key) == len(self.variables)
        idx = 0
        for k, c in zip(key, self.card, strict=True):
            idx = idx * c + k
        return self.values[idx]

    def to_dict(self) -> dict[tuple, float]:
        """All assignments → value (useful for testing / debugging)."""
        result: dict[tuple, float] = {}
        for assignment in _assignments(
            self.variables, {v: self.card_of(v) for v in self.variables}
        ):
            result[assignment] = self[assignment]
        return result


# -- module helpers -----------------------------------------------------------


def _flatten(nested: Sequence) -> list:
    """Recursively flatten nested sequences of numbers into a 1-D list."""
    result: list = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _prod(seq: Sequence[int]) -> int:
    p = 1
    for x in seq:
        p *= x
    return p


def _union_vars(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """Ordered union preserving *a*'s order, then appending new vars from *b*."""
    result = list(a)
    for v in b:
        if v not in result:
            result.append(v)
    return result


def _project(
    assignment: tuple[int, ...],
    variables: Sequence[str],
    union_lookup: dict[str, int],
) -> tuple[int, ...]:
    """Project a full union assignment down to the subset *variables*."""
    return tuple(assignment[union_lookup[v]] for v in variables)


def _assignments(
    variables: Sequence[str],
    cards: dict[str, int],
) -> list[tuple[int, ...]]:
    """Generate every assignment ``(v0_val, v1_val, ...)`` in variable order."""
    ranges = [range(cards[v]) for v in variables]
    return list(itertools.product(*ranges))
