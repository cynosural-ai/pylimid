"""
NumPyro influence-diagram solver — the intervention scan.

The unbatched reference implementation: solves a mixed-type influence
diagram by Monte-Carlo, enumerating the discrete policy space (one action
per information-set assignment per decision) and evaluating each policy by
forward-simulating the diagram with every decision resolved from its
observed information set, keeping the policy with the highest estimated
expected utility. The batched scan (batched_scan.py) is the same algorithm
with all policies evaluated in one vmapped call.

The forward pass mirrors to_model's topological walk (parents before
children — guaranteed by the Snapshot), except that a decision node is
not a sample site: its action is looked up from the candidate policy as a
deterministic function of the realized information-set values. Utility
nodes are evaluated on the sampled values after the walk; their average
over the samples is the expected-utility estimate.

Scope: decisions and their information sets must be discrete (a
continuous information set cannot be tabulated). Chance nodes may be
discrete or continuous. The policy space is exponential in the number of
decision rules, so this engine suits diagrams with small decision spaces.

References:
    Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving
    decision problems with limited information. Management Science
    47(9), 1235-1251. The LIMID formalism this solver targets and
    single policy updating, the efficient alternative to policy
    enumeration.

    Shachter, R. D. (1986). Evaluating influence diagrams. Operations
    Research 34(6), 871-882. Backward induction for influence diagrams,
    the exact counterpart to this scan's full enumeration.

    Bielza, C., Muller, P. and Rios Insua, D. (2007). Decision analysis
    by augmented probability simulation. Management Science 53(7).
    Monte-Carlo methods for decision analysis, the tradition this
    sampling-based scan belongs to.
"""

from __future__ import annotations

import jax

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference.numpyro.solvers._policy import (
    Rule,
    info_assignments,
    policy_array,
    policy_space,
)
from decisionpy.inference.result import Policy, Solution

__all__ = ["solve"]


def solve(
    snapshot: Snapshot,
    *,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve *snapshot* by the intervention scan.

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

    rules: list[Rule] = []
    for name, node in decisions:
        assignments = info_assignments(node, node_map)
        rules.append((name, node, assignments))

    best: tuple[float, Policy] | None = None
    for policy in policy_space(rules):
        expected_utility = _estimate_eu(
            snapshot, policy, num_samples, rng_key, node_map
        )
        if best is None or expected_utility > best[0]:
            best = (expected_utility, policy)

    assert best is not None
    return Solution(policy=best[1], expected_utility=best[0])


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
        name: policy_array(node, policy[name], node_map)
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
