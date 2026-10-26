"""
Internal translation of validated snapshots and bound rules into NumPyro models.

Chance nodes become sample sites, supplied deterministic policies become
deterministic sites, and fitted categorical policies become sample sites. Utilities
are evaluated by the sampling layer after drawing their parents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pylimid._numpyro.distributions import is_zero_count, validate_chance_distribution
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.nodes import ChanceNode, DecisionNode, UtilityNode
from pylimid.policy import CategoricalPolicy
from pylimid.policy._binding import BoundDecisionRule


def build_model(
    snapshot: DiagramSnapshot,
    observed: dict[str, Any],
    policy_rules: dict[str, BoundDecisionRule],
) -> Callable[[], dict[str, Any]]:
    """
    Build a NumPyro model using a validated snapshot and bound decision rules.

    The model walks nodes in topological order and resolves parents from upstream
    values. Unobserved discrete distributions that support enumeration carry
    parallel-enumeration metadata; observed chance nodes and fixed decisions are
    clamped. Numerical counts permitting only zero are clamped exactly. Utilities
    are evaluated separately by the sampling layer.

    Args:
        snapshot: A validated, topologically ordered diagram snapshot.
        observed: Observed chance values and fixed decision actions, keyed by name.
        policy_rules: Validated functions or fitted categorical rules keyed by
            decision name. Functions receive declared parents by keyword.

    Returns:
        A model function taking no arguments and returning node names mapped to
        their sampled, observed, or deterministic values.

    Raises:
        ValueError: If a chance distribution conflicts with its declared states
            when the returned model executes.

    Note:
        Enumeration may add dimensions to parent values. Index multi-parent
        probability tables as ``T[a, b]`` rather than ``T[a][b]`` to preserve
        NumPyro's posterior weights.
    """
    nodes = snapshot.nodes

    def model() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                if name in policy_rules:
                    parent_values = {parent: values[parent] for parent in node.parents}
                    rule = policy_rules[name]
                    if isinstance(rule, CategoricalPolicy):
                        values[name] = numpyro.sample(
                            name, rule._distribution(parent_values)
                        )
                    else:
                        value = rule(**parent_values)
                        values[name] = numpyro.deterministic(name, value)
                else:
                    # A fixed decision has all its probability at the supplied action.
                    value = observed[name]
                    values[name] = numpyro.sample(name, dist.Delta(value), obs=value)
                continue
            assert isinstance(node, ChanceNode)
            parent_values = {parent: values[parent] for parent in node.parents}
            assert node.dist is not None
            fn = node.dist(**parent_values)
            validate_chance_distribution(node, fn)
            if name in observed:
                values[name] = numpyro.sample(name, fn, obs=observed[name])
            elif not node.is_categorical and is_zero_count(fn):
                # Counts such as Binomial(total_count=0) can only be zero. Clamp
                # that guaranteed value because NumPyro cannot recover enumerated
                # counts with a single possible outcome. This adds no new evidence.
                value = jnp.zeros(fn.batch_shape, dtype=jnp.int32)
                values[name] = numpyro.sample(name, fn, obs=value)
            elif fn.support.is_discrete and fn.has_enumerate_support:
                # Ask inference to sum over the possible values of this unknown node.
                # Forward sampling ignores this annotation and draws normally.
                values[name] = numpyro.sample(name, fn, infer={"enumerate": "parallel"})
            else:
                values[name] = numpyro.sample(name, fn)
        return values

    return model
