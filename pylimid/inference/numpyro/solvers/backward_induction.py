"""
NumPyro influence-diagram solver — backward induction.

Solves a solvable (regular) influence diagram one decision at a time, in the
reverse order: the decisions closest to the utilities are resolved first, and
each resolved rule is used while estimating the previous decision. The
expected utility of every (information-set assignment, action) pair is
estimated by stratified forward sampling — trajectories are grouped by the
realized assignment and the utility is averaged within each group — so the
cost is additive in the decisions instead of the scan's exponential policy
product.

The gate is `solvability_order` (see `regularity`): a diagram without a
valid order raises instead of guessing. The non-solvable fallback is the
scan.

The RNG stream is a fixed seed per solve: the same ``rng_key`` and
``num_samples`` reproduce the same policy. Utilities ``values`` callables
must be JAX-traceable (no ``float()`` / ``int()`` coercion) — they are evaluated
inside the vmapped walk, as in the batched scan.

References:
    Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving
    decision problems with limited information. Management Science
    47(9), 1235-1251. Backward induction and single policy updating.

    Shachter, R. D. (1986). Evaluating influence diagrams. Operations
    Research 34(6), 871-882. The classical backward-induction algorithm.

    Bielza, C., Muller, P. and Rios Insua, D. (1999). Decision analysis
    by augmented probability simulation. Management Science 45(7),
    995-1007. Monte-Carlo estimation of the continuation values.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.utility_node import UtilityNode
from pylimid.inference.numpyro.solvers._policy import info_assignments
from pylimid.inference.numpyro.solvers.regularity import solvability_order
from pylimid.inference.solution import Policy, Solution

__all__ = ["solve"]


def solve(
    snapshot: DiagramSnapshot,
    *,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve *snapshot* by backward induction.

    Resolves the decisions in reverse order (closest to the utilities
    first), estimating every action's continuation value per
    information-set assignment by stratified forward sampling, and keeps
    the per-assignment argmax. The reported expected utility is a final
    forward pass with the fully resolved policy.

    Args:
        snapshot: A validated, solvable influence-diagram snapshot with at
            least one decision.
        num_samples: Forward trajectories per action per decision.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` whose policy maps every decision to an action per
        information-set assignment; the expected utility is a Monte-Carlo
        estimate.

    Raises:
        ValueError: If the diagram is not solvable (no backward-induction
            order — use the scan), or if a decision's information set is
            continuous, or if a decision's assignment never appeared in
            the samples (increase *num_samples*).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    order = solvability_order(snapshot)
    if order is None:
        raise ValueError(
            "The diagram is not solvable: no backward-induction order exists. "
            "Use the intervention scan, or add the memory arcs that make the "
            "decisions observe each other."
        )

    node_map = {
        name: node
        for name, node in snapshot.nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    decisions = {
        name: node for name, node in snapshot.nodes if isinstance(node, DecisionNode)
    }

    key = rng_key
    rules: dict[str, jax.Array] = {}
    policy: Policy = {}
    for name in order:
        key, subkey = jax.random.split(key)
        rule, sub_policy = _resolve_decision(
            snapshot, decisions[name], rules, num_samples, subkey, node_map
        )
        rules[name] = rule
        policy[name] = sub_policy

    key, subkey = jax.random.split(key)
    expected_utility = _estimate_policy_eu(snapshot, rules, num_samples, subkey)
    return Solution(
        policy=policy, expected_utility=expected_utility, method="backward_induction"
    )


# ---------------------------------------------------------------------------
# Decision resolution
# ---------------------------------------------------------------------------


def _resolve_decision(
    snapshot: DiagramSnapshot,
    decision: DecisionNode,
    rules: dict[str, jax.Array],
    num_samples: int,
    rng_key: jax.Array,
    node_map: dict[str, ChanceNode | DecisionNode],
):
    """Estimate the decision's Q-values and return (rule array, sub-policy)."""
    assignments = info_assignments(decision, node_map)
    cards: list[int] = []
    for parent in decision.parents:
        states = node_map[parent].states
        assert states is not None  # info_assignments rejected continuous info
        cards.append(len(states))
    assert decision.states is not None  # decisions always declare states
    step = _trajectory(snapshot, rules, target=decision)
    actions = jnp.arange(len(decision.states), dtype=jnp.int32)
    keys = jax.random.split(rng_key, num_samples)
    n_groups = len(assignments)

    @jax.jit
    def estimate(actions: jax.Array, keys: jax.Array):
        def per_action(action: jax.Array):
            return jax.vmap(step, in_axes=(None, 0))(action, keys)

        totals, groups = jax.vmap(per_action)(actions)
        one_hot = jax.nn.one_hot(groups, n_groups)
        counts = jnp.sum(one_hot, axis=1)
        sums = jnp.sum(totals[:, :, None] * one_hot, axis=1)
        quality = jnp.where(counts > 0, sums / jnp.maximum(counts, 1.0), 0.0)
        return quality, counts

    quality, counts = estimate(actions, keys)
    if not bool((counts > 0).all()):
        raise ValueError(
            f"Some information-set assignments of decision {decision.name!r} "
            f"were never sampled in {num_samples} trajectories; increase "
            f"num_samples."
        )

    rule_flat = jnp.argmax(quality, axis=0)
    rule = rule_flat.reshape(tuple(cards))
    sub_policy = {
        assignment: int(rule_flat[index])
        for index, assignment in enumerate(assignments)
    }
    return rule, sub_policy


def _estimate_policy_eu(
    snapshot: DiagramSnapshot,
    rules: dict[str, jax.Array],
    num_samples: int,
    rng_key: jax.Array,
) -> float:
    """The expected utility of the fully resolved policy."""
    step = _trajectory(snapshot, rules, target=None)
    keys = jax.random.split(rng_key, num_samples)

    @jax.jit
    def estimate(keys: jax.Array):
        totals, _ = jax.vmap(step, in_axes=(None, 0))(jnp.int32(0), keys)
        return jnp.mean(totals)

    return float(estimate(keys))


# ---------------------------------------------------------------------------
# Trajectory walk
# ---------------------------------------------------------------------------


def _trajectory(
    snapshot: DiagramSnapshot,
    rules: dict[str, jax.Array],
    target: DecisionNode | None,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """
    One forward trajectory with the decisions resolved so far.

    The *target* decision (when given) is clamped to the action passed to
    the returned step function; decisions in *rules* are resolved from
    their rule arrays; the remaining decisions act under a uniform
    placeholder. Returns the trajectory's total utility and — when a
    target is given — the target's flattened information-set assignment.
    """
    nodes = snapshot.nodes
    node_map = {
        name: node
        for name, node in nodes
        if isinstance(node, (ChanceNode, DecisionNode))
    }
    chance_nodes = [name for name, node in nodes if isinstance(node, ChanceNode)]
    utility_nodes = [
        (name, node) for name, node in nodes if isinstance(node, UtilityNode)
    ]
    target_name = target.name if target is not None else None
    unresolved = [
        name
        for name, node in nodes
        if isinstance(node, DecisionNode) and name not in rules and name != target_name
    ]
    n_random = len(chance_nodes) + len(unresolved)

    parents: tuple[str, ...] = ()
    cards: tuple[int, ...] = ()
    if target is not None:
        parents = target.parents
        parent_cards: list[int] = []
        for parent in parents:
            states = node_map[parent].states
            assert states is not None  # decisions only observe discrete nodes
            parent_cards.append(len(states))
        cards = tuple(parent_cards)

    def step(action: jax.Array, key: jax.Array):
        node_keys = jax.random.split(key, n_random)
        values: dict[str, jax.Array] = {}
        ki = 0
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                if name == target_name:
                    values[name] = action
                elif name in rules:
                    arr = rules[name]
                    values[name] = (
                        arr[tuple(values[p] for p in node.parents)]
                        if node.parents
                        else arr
                    )
                else:
                    assert node.states is not None  # decisions declare states
                    values[name] = jax.random.randint(
                        node_keys[ki], (), 0, len(node.states)
                    )
                    ki += 1
                continue
            assert isinstance(node, ChanceNode)
            assert node.dist is not None
            parent_values = {p: values[p] for p in node.parents}
            values[name] = node.dist(**parent_values).sample(node_keys[ki])
            ki += 1

        total = 0.0
        for _, node in utility_nodes:
            assert node.values is not None
            total = total + node.values(**{p: values[p] for p in node.parents})

        if not parents:
            return total, jnp.int32(0)
        index = jnp.int32(0)
        for parent, card in zip(parents, cards, strict=True):
            index = index * card + values[parent]
        return total, index

    return step
