"""Shared policy scenarios for solver and backend tests."""

import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.nodes import ChanceNode, DecisionNode, UtilityNode
from pylimid.policy import CategoricalPolicy, LinearSoftmax


def signal_diagram():
    """Build a linear decision rule over a scalar continuous signal."""
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode("signal", dist=lambda: dist.Uniform(-1.0, 1.0)))
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("signal",),
            states=("off", "on"),
            policy_family=LinearSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("signal", "action"),
            values=lambda signal, action: signal * action,
        )
    )
    return diagram


def uniform_rule(parents=(), parent_states=(), num_features=0):
    """
    Create a uniform linear rule for the supplied information metadata.

    Args:
        parents: Information-parent names in declared order.
        parent_states: Categorical labels, or ``None`` for numerical parents.
        num_features: Number of encoded information features.

    Returns:
        A categorical rule with zero weights and bias.
    """
    return CategoricalPolicy(
        LinearSoftmax(),
        parents,
        parent_states,
        tuple(() for _ in parents),
        ("off", "on"),
        {"weights": jnp.zeros((num_features, 2)), "bias": jnp.zeros(2)},
    )


def xor_diagram(family):
    """
    Build a binary decision rewarded for matching the XOR of its parents.

    Args:
        family: Trainable family assigned to the decision.

    Returns:
        A diagram with two binary signals and a decision-dependent payoff.
    """
    diagram = InfluenceDiagram()
    for name in ("left", "right"):
        diagram.add_node(
            ChanceNode(name, states=("no", "yes"), dist=lambda: dist.Bernoulli(0.5))
        )
    diagram.add_node(
        DecisionNode(
            "action",
            parents=("left", "right"),
            states=("off", "on"),
            policy_family=family,
        )
    )
    diagram.add_node(
        UtilityNode(
            "payoff",
            parents=("left", "right", "action"),
            values=lambda left, right, action: jnp.where(
                action == (left != right), 1.0, 0.0
            ),
        )
    )
    return diagram
