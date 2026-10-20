"""
NumPyro model factory — translate a validated snapshot into a NumPyro model.

`to_model` consumes a `DiagramSnapshot` and returns a plain NumPyro model function
the caller feeds to ``numpyro.infer``
(e.g. ``Predictive``, ``MCMC``, ``SVI``).

Unobserved discrete nodes with NumPyro enumeration support are detected automatically
and annotated with ``infer={"enumerate": "parallel"}`` so that NumPyro's NUTS
and SVI engines can marginalise them out. `samples` checks that numerical count
latents are scalar and have zero-based integer support. NumPyro requires concrete
enumeration bounds during JAX tracing. Observed nodes skip enumeration; numerical
distributions permitting only zero are clamped to that value.

Fixed decisions become degenerate observed sites; contingent decisions are
deterministic functions of their information sets. Utility nodes are skipped
— a payoff is never sampled and is never a parent.

Forward / prior-predictive sampling: pass ``observed=None`` (the default) and
feed the model to ``Predictive``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pylimid._distributions import _is_zero_count, _validate_chance_distribution
from pylimid.graph.chance_node import ChanceNode
from pylimid.graph.decision_node import DecisionNode
from pylimid.graph.diagram import DiagramSnapshot
from pylimid.graph.utility_node import UtilityNode
from pylimid.policy import Policy, _policy_rules

__all__ = ["to_model"]


def to_model(
    snapshot: DiagramSnapshot,
    observed: dict[str, Any] | None = None,
    policy: Policy | None = None,
) -> Callable[[], dict[str, Any]]:
    """
    Build a NumPyro model function from a validated snapshot.

    The returned ``model()`` walks the snapshot's nodes in topological order
    (parents before children — guaranteed by the snapshot), resolves each node's
    parents by name against the upstream sampled values, and emits one
    ``numpyro.sample`` site per chance node.

    Unobserved discrete chance nodes with NumPyro enumeration support are annotated with
    ``infer={"enumerate": "parallel"}`` so that NUTS and SVI can marginalise
    them out. Observed nodes skip enumeration and are clamped via ``obs=``.
    Continuous nodes and counts without enumeration support are left for the inference
    engine. Numerical distributions permitting only zero are clamped exactly.
    `samples` accepts only scalar numerical count latents with a fixed zero-based
    integer range, such as Binomial with a Python integer ``total_count``. Callers
    using this model directly with NumPyro must respect those enumeration limits;
    support metadata alone does not establish compatible bounds or returned values.

    A fixed decision becomes a degenerate observed site. A state-dependent
    decision policy is evaluated from its information-set parents as the
    model walks the diagram. A utility node is skipped entirely.

    Note:
        Under NumPyro's parallel enumeration, the parent values passed to a
        ``dist`` callable may carry an enumeration dimension. Index
        multi-parent probability tables in a single indexing operation
        (``T[a, b]``) rather than chained (``T[a][b]``) — NumPyro computes
        wrong posterior weights for chained indexing.

    Args:
        snapshot: A validated, topologically-ordered view of an influence
            diagram.
        observed: Mapping of node names to observed values. Chance nodes
            present here are conditioned on their given value; chance nodes
            absent are treated as latent. Fixed decisions are bound here.
        policy: Decision names mapped to complete action tables or deterministic
            functions. Functions accept declared information parents by keyword and
            return a scalar integer action index in the decision's ``states``.
            They must be JAX-traceable and currently support forward evaluation only.
            `samples` additionally checks sampled function actions against the action
            space; direct NumPyro callers must respect that output contract.

    Returns:
        A NumPyro model function taking no arguments, returning a
        ``dict[str, Any]`` of sampled (or observed) values keyed by node name.

    Raises:
        InferenceError: If a policy names a non-decision node, omits an
            information-set assignment, or contains an invalid action.
        KeyError: If a policy names a node that is not in the snapshot.
        ValueError: If a policy has an information-set parent without named states, or
            a chance distribution conflicts with its ``states`` when the model executes.
        NotImplementedError: If function policies are combined with chance-node
            evidence.
    """
    observed = observed or {}
    rules = _policy_rules(snapshot, policy or {}, observed)
    return _to_model(snapshot, observed, rules)


def _to_model(
    snapshot: DiagramSnapshot,
    observed: dict[str, Any],
    policy_rules: dict[str, Callable[..., jax.Array]],
) -> Callable[[], dict[str, Any]]:
    """Build a model using already validated decision rules."""
    nodes = snapshot.nodes

    def model() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, node in nodes:
            if isinstance(node, UtilityNode):
                continue
            if isinstance(node, DecisionNode):
                if name in policy_rules:
                    parent_values = {parent: values[parent] for parent in node.parents}
                    value = policy_rules[name](**parent_values)
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
            _validate_chance_distribution(node, fn)
            if name in observed:
                values[name] = numpyro.sample(name, fn, obs=observed[name])
            elif not node.is_categorical and _is_zero_count(fn):
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
