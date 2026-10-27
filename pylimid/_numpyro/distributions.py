"""Validate chance-node declarations against constructed NumPyro distributions."""

import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.discrete import (
    BernoulliLogits,
    BernoulliProbs,
    CategoricalLogits,
    CategoricalProbs,
)

from pylimid.graph.nodes import ChanceNode


def supports_count_enumeration(fn: Distribution) -> bool:
    """
    Check whether a count advertises zero-based integer-interval enumeration.

    Args:
        fn: Constructed NumPyro distribution to inspect.

    Returns:
        Whether enumeration is supported over an integer interval starting at zero.
    """
    support = fn.support
    return (
        fn.has_enumerate_support
        and isinstance(support, constraints.integer_interval)
        and bool(support.lower_bound == 0)
    )


def is_zero_count(fn: Distribution) -> bool:
    """
    Check whether concrete integer support permits only zero in every batch element.

    Args:
        fn: Constructed NumPyro distribution to inspect.

    Returns:
        Whether both integer-interval bounds are concrete and zero throughout.
    """
    support = fn.support
    return (
        isinstance(support, constraints.integer_interval)
        and not isinstance(support.lower_bound, Tracer)
        and not isinstance(support.upper_bound, Tracer)
        and bool(np.all(np.asarray(support.lower_bound) == 0))
        and bool(np.all(np.asarray(support.upper_bound) == 0))
    )


def validate_chance_distribution(node: ChanceNode, fn: Distribution) -> None:
    """
    Check distribution kind and static outcome count before sampling a node.

    Args:
        node: Chance node declaring categorical labels or numerical outcomes.
        fn: Distribution constructed from the node's parent values.

    Raises:
        ValueError: If categorical outcomes lack matching labels, a labeled node
            returns a numerical distribution, or numerical discrete events are
            non-scalar.
    """
    distribution = type(fn).__name__
    if node.states is None:
        if isinstance(
            fn, (CategoricalProbs, CategoricalLogits, BernoulliProbs, BernoulliLogits)
        ):
            raise ValueError(
                f"Chance node {node.name!r} has states=None but returns discrete "
                f"{distribution}. Categorical and Bernoulli distributions require "
                "matching states for categorical outcomes."
            )
        if fn.support.is_discrete and fn.event_shape:  # ty: ignore[possibly-missing-attribute]
            raise ValueError(
                f"Chance node {node.name!r} returns {distribution} with event shape "
                f"{fn.event_shape}. Numerical discrete distributions require scalar "
                "events."
            )
        return

    if isinstance(fn, CategoricalProbs):
        outcomes = jnp.shape(fn.probs)[-1]
    elif isinstance(fn, CategoricalLogits):
        outcomes = jnp.shape(fn.logits)[-1]
    elif isinstance(fn, (BernoulliProbs, BernoulliLogits)):
        outcomes = 2
    else:
        raise ValueError(
            f"Chance node {node.name!r} declares states but returns {distribution}. "
            "Labeled chance nodes require Categorical or Bernoulli with one label "
            "per outcome. Numerical distributions require states=None."
        )

    if len(node.states) != outcomes:
        raise ValueError(
            f"Chance node {node.name!r} has {len(node.states)} state labels, but "
            f"{distribution} has {outcomes} outcomes; states must contain exactly "
            f"{outcomes} labels, including zero-probability outcomes."
        )
