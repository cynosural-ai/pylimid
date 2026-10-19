"""Policies and their shared state-indexed array representation."""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

import jax
import jax.numpy as jnp

from pylimid.errors import InferenceError
from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot

__all__ = ["Policy"]

#: Policy tables: decision name -> {parent-state indices: action index}.
Policy: TypeAlias = dict[str, dict[tuple[int, ...], int]]


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
    sub_policy: dict[tuple[int, ...], int],
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


def _policy_tables(snapshot: DiagramSnapshot, policy: Policy) -> dict[str, jax.Array]:
    """Validate contingent rules and encode them as state-indexed arrays."""
    nodes = dict(snapshot.nodes)
    node_map = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    tables = {}
    for name, rule in policy.items():
        decision = nodes[name]
        if not isinstance(decision, DecisionNode):
            raise InferenceError(f"Policy for {name!r} must bind a decision node.")
        if set(rule) != set(_info_assignments(decision, node_map)):
            raise InferenceError(
                f"Policy for {name!r} must define every information-set assignment."
            )
        assert decision.states is not None
        if any(
            not isinstance(action, int) or not 0 <= action < len(decision.states)
            for action in rule.values()
        ):
            raise InferenceError(f"Policy for {name!r} contains an invalid action.")
        tables[name] = _policy_array(decision, rule, node_map)
    return tables
