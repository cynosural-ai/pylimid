"""Enumerate candidate policies and encode batches for the scan solvers."""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

import jax
import jax.numpy as jnp

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.policy import Policy

__all__ = ["Rule", "batched_policy_array", "policy_space"]

#: A decision's rule space: its name, node, and every information-set assignment.
Rule: TypeAlias = tuple[str, DecisionNode, list[tuple[int, ...]]]


def policy_space(rules: list[Rule]):
    """
    Yield every policy over the decision rules.

    One action per information-set assignment per decision, in decision
    (topological) order.
    """
    action_rules = []
    for _, node, assignments in rules:
        assert node.states is not None  # decisions always declare states
        action_rules.append(product(range(len(node.states)), repeat=len(assignments)))
    for combination in product(*action_rules):
        policy: Policy = {}
        for (name, _node, assignments), actions in zip(rules, combination, strict=True):
            policy[name] = dict(zip(assignments, actions, strict=True))
        yield policy


def batched_policy_array(
    decision: DecisionNode,
    sub_policies: list[dict[tuple[int, ...], int]],
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """
    All of *decision*'s sub-policies as one array over (policy, info set).

    The leading axis indexes *sub_policies*; the remaining axes are
    indexed by the information-set assignment, with the same axes for each policy.
    """
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # information sets have been checked
        cards.append(len(states))
    arr = jnp.zeros((len(sub_policies), *cards), dtype=jnp.int32)
    for i, sub_policy in enumerate(sub_policies):
        for assignment, action in sub_policy.items():
            arr = arr.at[(i,) + assignment].set(action)
    return arr
