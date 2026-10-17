"""Validate chance-node declarations against constructed NumPyro distributions."""

import jax.numpy as jnp
from numpyro.distributions import Distribution
from numpyro.distributions.discrete import (
    BernoulliLogits,
    BernoulliProbs,
    CategoricalLogits,
    CategoricalProbs,
)

from pylimid.graph.chance_node import ChanceNode


def _validate_chance_distribution(node: ChanceNode, fn: Distribution) -> None:
    """Check distribution kind and static outcome count before sampling a node."""
    distribution = type(fn).__name__
    if node.states is None:
        if fn.support.is_discrete:  # ty: ignore[possibly-missing-attribute]
            raise ValueError(
                f"Chance node {node.name!r} has states=None but returns discrete "
                f"{distribution}. Unlabeled chance nodes require a continuous "
                "distribution; use Categorical or Bernoulli with matching states "
                "for categorical outcomes."
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
            "per outcome. Continuous distributions require states=None."
        )

    if len(node.states) != outcomes:
        raise ValueError(
            f"Chance node {node.name!r} has {len(node.states)} state labels, but "
            f"{distribution} has {outcomes} outcomes; states must contain exactly "
            f"{outcomes} labels, including zero-probability outcomes."
        )
