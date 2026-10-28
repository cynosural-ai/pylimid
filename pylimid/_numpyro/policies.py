"""Adapt fitted policies to NumPyro's leading enumeration and batch dimensions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.batch_util import promote_batch_shape

from pylimid.policy import CategoricalPolicy, DecisionInformation


def policy_distribution(
    rule: CategoricalPolicy, observations: DecisionInformation
) -> dist.CategoricalLogits | dist.CategoricalProbs:
    """
    Construct an action distribution over broadcast information assignments.

    Each parent's trailing dimensions retain its fitted value shape. Leading
    dimensions broadcast across parents and index separate information assignments;
    each assignment is evaluated through the rule's single-assignment interface.

    Args:
        rule: Fitted categorical policy with parent value shapes.
        observations: Information-parent values, optionally with leading enumeration
            or batch dimensions.

    Returns:
        A categorical distribution with the broadcast leading dimensions as its
        batch shape and probabilities in the rule's action order. The family's
        probability or logit parameterization is preserved.

    Raises:
        TypeError: If observations include an undeclared information parent.
        KeyError: If a declared information parent is missing.
        ValueError: If parent values do not retain their fitted shapes, leading
            dimensions cannot broadcast, or the rule returns an incompatible action
            distribution.
    """
    unexpected = observations.keys() - rule.parents
    if unexpected:
        raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")

    shapes = dict(zip(rule.parents, rule.parent_shapes, strict=True))
    values: dict[str, jax.Array] = {}
    batch_shape: tuple[int, ...] = ()
    for parent, shape in shapes.items():
        value = jnp.asarray(observations[parent])
        batch_ndim = value.ndim - len(shape)
        if batch_ndim < 0 or value.shape[batch_ndim:] != shape:
            raise ValueError(
                f"Information parent {parent!r} must have trailing shape {shape}."
            )
        batch_shape = jnp.broadcast_shapes(batch_shape, value.shape[:batch_ndim])
        values[parent] = value

    if not batch_shape:
        return rule._distribution(values)

    values = {
        parent: jnp.broadcast_to(value, batch_shape + shapes[parent])
        for parent, value in values.items()
    }

    evaluate = rule._distribution
    for _ in batch_shape:
        evaluate = jax.vmap(evaluate)
    # vmap batches parameter arrays but preserves single-assignment batch metadata.
    return promote_batch_shape(evaluate(values))
