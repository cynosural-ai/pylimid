"""
Influence-diagram solver — the batched intervention scan.

Same algorithm and same semantics as
``pylimid.solver.scan.intervention``, evaluated in one
vmapped, jitted pass: every policy array carries a leading batch
dimension, and the forward walk is nested-vmapped over (policy, sample).
One trace, one dispatch for the whole solve — the per-policy re-trace
overhead of the unbatched scan disappears.

The RNG stream matches the unbatched scan exactly: the same per-sample
keys are broadcast across the policy axis, so both solvers draw identical
samples and select the same optimal policy; the expected utilities agree
to float rounding (the unbatched scan accumulates in float64 in Python,
this one in the JAX dtype).

Constraint: utility ``values`` callables must be JAX-traceable (pure ``jnp``
expressions — no ``float()`` / ``int()`` coercion), because they are evaluated
inside the vmapped walk.

References:
    Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving
    decision problems with limited information. Management Science
    47(9), 1235-1251. The LIMID formalism this solver targets and
    single policy updating, the efficient alternative to policy
    enumeration.

    Shachter, R. D. (1986). Evaluating influence diagrams. Operations
    Research 34(6), 871-882. Backward induction for influence diagrams,
    the exact counterpart to this scan's full enumeration.

    Bielza, C., Muller, P. and Rios Insua, D. (1999). Decision analysis
    by augmented probability simulation. Management Science 45(7),
    995-1007. Monte-Carlo methods for decision analysis, the tradition this
    sampling-based scan belongs to.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from pylimid._distributions import _validate_chance_distribution
from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.utility_node import UtilityNode
from pylimid.policy import Policy, _info_assignments, _require_tabular_families
from pylimid.solver.scan._policy import (
    Rule,
    batched_policy_array,
    policy_space,
)
from pylimid.solver.solution import Solution

__all__ = ["solve"]


def solve(
    snapshot: DiagramSnapshot,
    *,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve *snapshot* by the batched intervention scan.

    Enumerates every policy, estimates its expected total utility by forward sampling,
    and selects the highest estimate. All policies are evaluated together with JAX. This
    is the scan used by ``pylimid.solve(method="scan")``; `scan_solve` evaluates the
    same policies individually. Their estimates agree up to floating-point rounding for
    the same ``rng_key`` and ``num_samples``.

    For most uses, call `pylimid.solve`, which validates the diagram and selects a
    solver. Chance distributions and utility functions must be JAX-traceable. The number
    of policies grows exponentially with the number of action-table entries.

    Args:
        snapshot: A validated snapshot from `InfluenceDiagram.snapshot`, with finite
            decisions and categorical information sets.
        num_samples: Forward samples per candidate policy.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` whose policy maps every decision to an action per
        information-set assignment; the expected utility is a Monte-Carlo
        estimate.

    Raises:
        ValueError: If a decision's information set contains a parent without named
            states, a decision declares a policy family, or a chance distribution
            conflicts with its ``states``.
    """
    _require_tabular_families(snapshot)
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
        assignments = _info_assignments(node, node_map)
        rules.append((name, node, assignments))

    policies = list(policy_space(rules))
    policy_arrays = {
        name: batched_policy_array(
            node, [policy[name] for policy in policies], node_map
        )
        for name, node in decisions
    }
    eu = _estimate_eu(snapshot, policy_arrays, num_samples, rng_key, node_map)
    best = int(jnp.argmax(eu))
    return Solution(
        policy=cast(Policy, policies[best]),
        expected_utility=float(eu[best]),
        method="scan",
    )


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def _estimate_eu(
    snapshot: DiagramSnapshot,
    policy_arrays: dict[str, jax.Array],
    num_samples: int,
    rng_key: jax.Array,
    node_map: dict[str, ChanceNode | DecisionNode],
) -> jax.Array:
    """Estimate E[U | policy] for every policy in one vmapped, jitted pass."""
    nodes = snapshot.nodes
    chance_nodes = [name for name, node in nodes if isinstance(node, ChanceNode)]
    utility_nodes = [
        (name, node) for name, node in nodes if isinstance(node, UtilityNode)
    ]

    @jax.jit
    def evaluate(arrays, keys):
        def eval_policy(arrays_per_policy, keys_per_policy):
            def step(key: jax.Array) -> dict[str, jax.Array]:
                node_keys = jax.random.split(key, len(chance_nodes))
                values: dict[str, jax.Array] = {}
                ki = 0
                for name, node in nodes:
                    if isinstance(node, UtilityNode):
                        continue
                    if isinstance(node, DecisionNode):
                        arr = arrays_per_policy[name]
                        values[name] = (
                            arr[tuple(values[p] for p in node.parents)]
                            if node.parents
                            else arr
                        )
                        continue
                    assert isinstance(node, ChanceNode)
                    assert node.dist is not None
                    parent_values = {p: values[p] for p in node.parents}
                    fn = node.dist(**parent_values)
                    _validate_chance_distribution(node, fn)
                    values[name] = fn.sample(node_keys[ki])
                    ki += 1
                return values

            sampled = jax.vmap(step)(keys_per_policy)
            total = 0.0
            for _, node in utility_nodes:
                values_fn = node.values
                assert values_fn is not None
                utility = jax.vmap(
                    lambda row, fn=values_fn, parents=node.parents: fn(
                        **{p: row[p] for p in parents}
                    )
                )(sampled)
                total = total + utility
            return jnp.mean(total, axis=0)

        return jax.vmap(eval_policy, in_axes=(0, None))(arrays, keys)

    # The same keys for every policy: identical draws to the unbatched
    # scan, which passes the same rng_key to each policy evaluation.
    keys = jax.random.split(rng_key, num_samples)
    return evaluate(policy_arrays, keys)
