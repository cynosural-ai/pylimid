"""
NumPyro influence-diagram solver — the batched intervention scan.

Same algorithm and same semantics as
`decisionpy.inference.numpyro.solvers.intervention_scan`, evaluated in one
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

import jax
import jax.numpy as jnp

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference.numpyro.solvers._policy import (
    Rule,
    batched_policy_array,
    info_assignments,
    policy_space,
)
from decisionpy.inference.result import Solution

__all__ = ["solve"]


def solve(
    snapshot: Snapshot,
    *,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve *snapshot* by the batched intervention scan.

    Enumerates the discrete policy space, estimates expected utility for
    every policy in one vmapped forward pass, and keeps the best policy.
    With the same *rng_key* and *num_samples*, returns the same policy as
    `intervention_scan.solve`; the expected utility matches to float
    rounding.

    Args:
        snapshot: A validated influence-diagram snapshot with at least one
            decision.
        num_samples: Forward samples per candidate policy.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``.

    Returns:
        A `Solution` whose policy maps every decision to an action per
        information-set assignment; the expected utility is a Monte-Carlo
        estimate.

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

    policies = list(policy_space(rules))
    policy_arrays = {
        name: batched_policy_array(
            node, [policy[name] for policy in policies], node_map
        )
        for name, node in decisions
    }
    eu = _estimate_eu(snapshot, policy_arrays, num_samples, rng_key, node_map)
    best = int(jnp.argmax(eu))
    return Solution(policy=policies[best], expected_utility=float(eu[best]))


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def _estimate_eu(
    snapshot: Snapshot,
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
                    values[name] = node.dist(**parent_values).sample(node_keys[ki])
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
