"""Validate and bind supplied decision policies to a diagram."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from typing import cast

import jax
import jax.numpy as jnp

from pylimid.errors import InferenceError
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.nodes.chance_node import ChanceNode
from pylimid.graph.nodes.decision_node import DecisionNode
from pylimid.graph.nodes.node import _signature_matches
from pylimid.policy.rules import CategoricalPolicy, Policy


def bind_policy_rules(
    snapshot: DiagramSnapshot,
    policy: Policy,
    observed: Mapping[str, object],
) -> dict[str, Callable[..., jax.Array] | CategoricalPolicy]:
    """
    Validate supplied policies and bind them to their diagram information sets.

    Args:
        snapshot: Validated snapshot containing the decisions to bind.
        policy: Decision names mapped to complete tables, deterministic functions,
            or fitted categorical rules. Fixed actions are supplied in ``observed``.
        observed: Observed chance values and fixed decision actions, keyed by name.

    Returns:
        Decision names mapped to functions or fitted categorical rules ready for
        evaluation with their declared information parents.

    Raises:
        InferenceError: If a rule binds a non-decision, has incompatible information
            or action labels, uses an incompatible callable signature, or supplies an
            incomplete table or invalid action.
        KeyError: If a policy names a node outside the snapshot.
        ValueError: If a table policy has a parent without named finite states.
        NotImplementedError: If functions or fitted rules are combined with
            observed chance-node evidence.
    """
    nodes = dict(snapshot.nodes)
    node_map = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    has_chance_evidence = any(
        name in observed and isinstance(node, ChanceNode)
        for name, node in snapshot.nodes
    )
    rules = {}
    for name, rule in policy.items():
        decision = nodes[name]
        if not isinstance(decision, DecisionNode):
            raise InferenceError(f"Policy for {name!r} must bind a decision node.")
        if isinstance(rule, CategoricalPolicy):
            if (
                set(rule.parents) != set(decision.parents)
                or rule.states != decision.states
                or any(
                    states != node_map[parent].states
                    for parent, states in zip(
                        rule.parents, rule.parent_states, strict=True
                    )
                )
            ):
                raise InferenceError(
                    f"Policy for {name!r} has incompatible information "
                    "or action states."
                )
            if has_chance_evidence:
                raise NotImplementedError(
                    "Fitted categorical policies support forward evaluation only."
                )
            rules[name] = rule
            continue
        if not isinstance(rule, dict):
            if not _signature_matches(rule, decision.parents):
                raise InferenceError(
                    f"Policy for {name!r} must accept its information parents "
                    f"{decision.parents!r} as named keyword arguments."
                )
            if has_chance_evidence:
                raise NotImplementedError(
                    "Function policies support forward evaluation only; "
                    "conditioning on chance-node evidence is not supported."
                )
            rules[name] = _function_rule(name, rule)
            continue
        rule = cast(dict[tuple[int, ...], int], rule)
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
        rules[name] = _table_rule(
            _policy_array(decision, rule, node_map), decision.parents
        )
    return rules


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


def _table_rule(table: jax.Array, parents: tuple[str, ...]) -> Callable[..., jax.Array]:
    """Wrap a compiled action table as a function of named parent values."""

    def evaluate(**parent_values: jax.Array) -> jax.Array:
        return table[tuple(parent_values[parent] for parent in parents)]

    return evaluate


def _function_rule(
    name: str, function: Callable[..., int | jax.Array]
) -> Callable[..., jax.Array]:
    """Require a function rule to return a scalar integer action."""

    def evaluate(**parent_values: jax.Array) -> jax.Array:
        action = jnp.asarray(function(**parent_values))
        if action.shape != () or not jnp.issubdtype(action.dtype, jnp.integer):
            raise InferenceError(
                f"Policy for {name!r} must return a scalar integer action index; "
                f"got shape {action.shape} and dtype {action.dtype}."
            )
        return action

    return evaluate
