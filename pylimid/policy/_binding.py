"""Validate and bind supplied decision policies to a diagram."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import jax
import jax.numpy as jnp

from pylimid._signatures import signature_matches
from pylimid.errors import InferenceError
from pylimid.graph import Interval
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.nodes import ChanceNode, DecisionNode
from pylimid.policy._categorical.action_table import bind_action_table
from pylimid.policy._categorical.policy import CategoricalPolicy
from pylimid.policy._continuous.policy import ContinuousPolicy
from pylimid.policy.types import ActionTable, DecisionRule

#: An execution-ready rule produced by `bind_policy_rules`. Tables become lookup
#: functions, and supplied functions are wrapped to check and convert their scalar
#: results to JAX arrays. Fitted `CategoricalPolicy` and `ContinuousPolicy` rules
#: pass through.
#: Both representations receive information-parent values by keyword.
type BoundDecisionRule = Callable[..., jax.Array] | CategoricalPolicy | ContinuousPolicy


def bind_policy_rules(
    snapshot: DiagramSnapshot,
    policy: Mapping[str, DecisionRule],
    observed: Mapping[str, object],
) -> dict[str, BoundDecisionRule]:
    """
    Validate supplied policies and bind them to their diagram information sets.

    Args:
        snapshot: Validated snapshot containing the decisions to bind.
        policy: Decision names mapped to complete tables, deterministic functions,
            or fitted stochastic rules. Fixed actions are supplied in ``observed``.
        observed: Observed chance values and fixed decision actions, keyed by name.

    Returns:
        Decision names mapped to functions or fitted stochastic rules ready for
        evaluation with their declared information parents.

    Raises:
        InferenceError: If a rule binds a non-decision, has incompatible information
            or action labels, uses an incompatible callable signature, or supplies an
            incomplete table or invalid action.
        KeyError: If a policy names a node outside the snapshot.
        ValueError: If a table policy has a parent without named finite states.
        NotImplementedError: If function policies are combined with observed
            chance-node evidence.
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
    rules: dict[str, BoundDecisionRule] = {}
    for name, node in snapshot.nodes:
        if isinstance(node, DecisionNode) and name in observed:
            validate_action_values(node, _scalar_action(node, observed[name]))
    for name, rule in policy.items():
        decision = nodes[name]
        if not isinstance(decision, DecisionNode):
            raise InferenceError(f"Policy for {name!r} must bind a decision node.")
        if isinstance(rule, (CategoricalPolicy, ContinuousPolicy)):
            rules[name] = _bind_fitted_policy(decision, rule, node_map)
        elif isinstance(rule, dict):
            rules[name] = bind_action_table(decision, cast(ActionTable, rule), node_map)
        else:
            rules[name] = _bind_function_policy(decision, rule, has_chance_evidence)
    return rules


def validate_action_values(decision: DecisionNode, actions: jax.Array) -> None:
    """
    Check supplied fixed or deterministic action values against their domain.

    Args:
        decision: Configured decision whose scalar action types were checked.
        actions: One action or a batch of actions with no event dimensions.

    Raises:
        InferenceError: If a supplied action is non-finite or outside its domain.
    """
    if decision.states is not None:
        invalid = (actions < 0) | (actions >= len(decision.states))
    else:
        invalid = ~jnp.isfinite(actions)
        if isinstance(decision.action_space, Interval):
            invalid = (
                invalid
                | (actions < decision.action_space.low)
                | (actions > decision.action_space.high)
            )
    if bool(jnp.any(invalid)):
        raise InferenceError(
            f"Policy for {decision.name!r} returned an invalid action."
        )


def _bind_fitted_policy(
    decision: DecisionNode,
    rule: CategoricalPolicy | ContinuousPolicy,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> CategoricalPolicy | ContinuousPolicy:
    """Check a fitted rule's information metadata and action domain against a node."""
    if isinstance(rule, CategoricalPolicy):
        compatible_actions = (
            decision.action_space is None and rule.states == decision.states
        )
    else:
        compatible_actions = rule.action_space == decision.action_space
    if (
        set(rule.parents) != set(decision.parents)
        or not compatible_actions
        or any(
            states != node_map[parent].states
            for parent, states in zip(rule.parents, rule.parent_states, strict=True)
        )
    ):
        raise InferenceError(
            f"Policy for {decision.name!r} has incompatible information "
            "or action states/domain."
        )
    return rule


def _bind_function_policy(
    decision: DecisionNode,
    function: Callable[..., int | float | jax.Array],
    has_chance_evidence: bool,
) -> Callable[..., jax.Array]:
    """Check function compatibility and wrap its scalar action result."""
    if not signature_matches(function, decision.parents):
        raise InferenceError(
            f"Policy for {decision.name!r} must accept its information parents "
            f"{decision.parents!r} as named keyword arguments."
        )
    if has_chance_evidence:
        raise NotImplementedError(
            "Function policies support forward evaluation only; "
            "conditioning on chance-node evidence is not supported."
        )

    def evaluate(**parent_values: jax.Array) -> jax.Array:
        return _scalar_action(decision, function(**parent_values))

    return evaluate


def _scalar_action(decision: DecisionNode, value: object) -> jax.Array:
    """Check the shape and numerical kind of one supplied deterministic action."""
    action = jnp.asarray(value)
    integer = jnp.issubdtype(action.dtype, jnp.integer)
    valid_type = (
        integer
        if decision.is_categorical
        else integer or jnp.issubdtype(action.dtype, jnp.floating)
    )
    if action.shape != () or not valid_type:
        description = (
            "a scalar integer action index"
            if decision.is_categorical
            else "a real-valued scalar action"
        )
        raise InferenceError(
            f"Policy for {decision.name!r} must return {description}; "
            f"got shape {action.shape} and dtype {action.dtype}."
        )
    return action
