"""Validate and bind deterministic categorical action tables."""

from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp

from pylimid.errors import InferenceError
from pylimid.graph.nodes import ChanceNode, DecisionNode
from pylimid.policy.types import ActionTable


def bind_action_table(
    decision: DecisionNode,
    table: ActionTable,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> Callable[..., jax.Array]:
    """
    Validate a complete categorical action table and bind its information parents.

    Args:
        decision: Decision whose categorical actions the table selects.
        table: Scalar parent-state assignments mapped to action indices.
        node_map: Chance and decision nodes keyed by name.

    Returns:
        A function accepting the declared information parents by keyword.

    Raises:
        InferenceError: If the decision is continuous, the table is incomplete, or
            an action is outside the decision's states.
        ValueError: If an information parent lacks named finite states.
    """
    if decision.action_space is not None:
        raise InferenceError(
            "ActionTable policies require categorical decision actions."
        )
    if set(table) != set(_info_assignments(decision, node_map)):
        raise InferenceError(
            f"Policy for {decision.name!r} must define "
            "every information-set assignment."
        )
    assert decision.states is not None
    if any(
        not isinstance(action, int) or not 0 <= action < len(decision.states)
        for action in table.values()
    ):
        raise InferenceError(
            f"Policy for {decision.name!r} contains an invalid action."
        )
    return _table_rule(_policy_array(decision, table, node_map), decision.parents)


def _info_assignments(
    decision: DecisionNode,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> list[tuple[int, ...]]:
    """
    Every assignment of the decision's information set, in parent order.

    Raises:
        ValueError: If an information-set parent has no named states.
    """
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        if states is None:
            raise ValueError(
                f"Decision {decision.name!r} has parent {parent!r} without named "
                "states in its information set; tabular policies require finite "
                "categorical information sets."
            )
        cards.append(len(states))
    return list(product(*[range(c) for c in cards]))


def _policy_array(
    decision: DecisionNode,
    sub_policy: ActionTable,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """The decision's sub-policy as an array indexed by the info assignment."""
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # _info_assignments requires named states
        cards.append(len(states))
    arr = jnp.zeros(tuple(cards), dtype=jnp.int32)
    for assignment, action in sub_policy.items():
        arr = arr.at[assignment].set(action)
    return arr


def _table_rule(table: jax.Array, parents: tuple[str, ...]) -> Callable[..., jax.Array]:
    """Wrap a compiled action table as a function of named parent values."""

    def evaluate(**parent_values: jax.Array) -> jax.Array:
        return table[tuple(parent_values[parent] for parent in parents)]

    return evaluate
