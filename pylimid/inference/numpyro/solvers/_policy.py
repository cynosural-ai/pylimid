"""
Policy-space representation shared by the solver family.

Internal module. The solvers differ in how they evaluate policies, but they
describe and encode them the same way: a policy is one action per
information-set assignment per decision, and a decision's sub-policies become
int32 arrays indexed by the parent-state assignment.
"""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

import jax
import jax.numpy as jnp

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.inference.result import Policy

__all__ = [
    "Rule",
    "batched_policy_array",
    "info_assignments",
    "policy_array",
    "policy_space",
]

#: A decision's rule space: its name, node, and every information-set
#: assignment in parent order.
Rule: TypeAlias = tuple[str, DecisionNode, list[tuple[int, ...]]]


def info_assignments(
    decision: DecisionNode,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> list[tuple[int, ...]]:
    """
    Every assignment of the decision's information set, in parent order.

    Raises:
        ValueError: If an information-set parent is continuous.
    """
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        if states is None:
            raise ValueError(
                f"Decision {decision.name!r} has continuous parent "
                f"{parent!r} in its information set; the NumPyro solvers "
                f"require discrete information sets."
            )
        cards.append(len(states))
    return list(product(*[range(c) for c in cards]))


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


def policy_array(
    decision: DecisionNode,
    sub_policy: dict[tuple[int, ...], int],
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """The decision's sub-policy as an array indexed by the info assignment."""
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # info_assignments rejected continuous info
        cards.append(len(states))
    arr = jnp.zeros(tuple(cards), dtype=jnp.int32)
    for assignment, action in sub_policy.items():
        arr = arr.at[assignment].set(action)
    return arr


def batched_policy_array(
    decision: DecisionNode,
    sub_policies: list[dict[tuple[int, ...], int]],
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """
    All of *decision*'s sub-policies as one array over (policy, info set).

    The leading axis indexes *sub_policies*; the remaining axes are
    indexed by the information-set assignment, mirroring policy_array per
    policy.
    """
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # info_assignments rejected continuous info
        cards.append(len(states))
    arr = jnp.zeros((len(sub_policies), *cards), dtype=jnp.int32)
    for i, sub_policy in enumerate(sub_policies):
        for assignment, action in sub_policy.items():
            arr = arr.at[(i,) + assignment].set(action)
    return arr
