"""
NumPyro influence-diagram solver — Strategy B, the intervention scan.

Solves a mixed-type influence diagram by Monte-Carlo: enumerate the
discrete policy space (one action per information-set assignment per
decision), evaluate each policy by forward-simulating the diagram with
every decision resolved from its observed information set, and keep the
policy with the highest estimated expected utility.

The forward pass mirrors to_model's topological walk (parents before
children — guaranteed by the Snapshot), except that a decision node is
not a sample site: its action is looked up from the candidate policy as a
deterministic function of the realized information-set values. Utility
nodes are evaluated on the sampled values after the walk; their average
over the samples is the expected-utility estimate.

Scope: decisions and their information sets must be discrete (a
continuous information set cannot be tabulated — that is the
policy-as-parameters path, not implemented yet). Chance nodes may be
discrete or continuous. The policy space is exponential in the number of
decision rules, so this engine suits diagrams with small decision spaces;
it is the Monte-Carlo counterpart to bucket elimination, which stays the
exact solver for all-categorical diagrams.
"""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference.result import Policy, Solution

__all__ = ["solve"]


def solve(
    snapshot: Snapshot,
    *,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve *snapshot* by intervention scan (Strategy B).

    Enumerates the discrete policy space, estimates expected utility per
    policy by forward sampling with each decision resolved from its
    observed information set, and keeps the best policy.

    Args:
        snapshot: A validated influence-diagram snapshot with at least one
            decision.
        num_samples: Forward samples per candidate policy.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A Solution whose policy maps every decision to an action per
        information-set assignment, with ``exact=False`` — the
        expected utility is a Monte-Carlo estimate.

    Raises:
        ValueError: If a decision's information set contains a continuous
            variable (a continuous information set cannot be tabulated).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    node_map = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    decisions = [
        (name, node) for name, node in snapshot.nodes if isinstance(node, DecisionNode)
    ]

    rules: list[tuple[str, DecisionNode, list[tuple[int, ...]]]] = []
    for name, node in decisions:
        assignments = _info_assignments(node, node_map)
        rules.append((name, node, assignments))

    best: tuple[float, Policy] | None = None
    for policy in _policy_space(rules):
        expected_utility = _estimate_eu(
            snapshot, policy, num_samples, rng_key, node_map
        )
        if best is None or expected_utility > best[0]:
            best = (expected_utility, policy)

    assert best is not None
    return Solution(
        policy=best[1],
        expected_utility=best[0],
        exact=False,
    )


# ---------------------------------------------------------------------------
# Policy space
# ---------------------------------------------------------------------------


def _info_assignments(
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
                f"{parent!r} in its information set; the intervention-scan "
                f"solver requires discrete information sets."
            )
        cards.append(len(states))
    return list(product(*[range(c) for c in cards]))


def _policy_space(
    rules: list[tuple[str, DecisionNode, list[tuple[int, ...]]]],
):
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


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def _estimate_eu(
    snapshot: Snapshot,
    policy: Policy,
    num_samples: int,
    rng_key: jax.Array,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> float:
    """Estimate E[U | policy] by forward sampling with decisions resolved."""
    nodes = snapshot.nodes
    decision_arrays = {
        name: _policy_array(node, policy[name], node_map)
        for name, node in nodes
        if isinstance(node, DecisionNode)
    }
    chance_nodes = [name for name, node in nodes if isinstance(node, ChanceNode)]
    utility_nodes = [
        (name, node) for name, node in nodes if isinstance(node, UtilityNode)
    ]

    def step(key: jax.Array) -> dict[str, jax.Array]:
        node_keys = jax.random.split(key, len(chance_nodes))
        values: dict[str, jax.Array] = {}
        ki = 0
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                arr = decision_arrays[name]
                values[name] = (
                    arr[tuple(values[p] for p in node.parents)] if node.parents else arr
                )
                continue
            assert isinstance(node, ChanceNode)
            assert node.dist is not None
            parent_values = {p: values[p] for p in node.parents}
            values[name] = node.dist(**parent_values).sample(node_keys[ki])
            ki += 1
        return values

    sampled = jax.vmap(step)(jax.random.split(rng_key, num_samples))
    total = 0.0
    for i in range(num_samples):
        row = {name: sampled[name][i] for name in sampled}
        for _, node in utility_nodes:
            assert node.values is not None
            total += float(node.values(**{p: row[p] for p in node.parents}))
    return total / num_samples


def _policy_array(
    decision: DecisionNode,
    sub_policy: dict[tuple[int, ...], int],
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """The decision's sub-policy as an array indexed by the info assignment."""
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # _info_assignments rejected continuous info
        cards.append(len(states))
    arr = jnp.zeros(tuple(cards), dtype=jnp.int32)
    for assignment, action in sub_policy.items():
        arr = arr.at[assignment].set(action)
    return arr
