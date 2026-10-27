"""Adapt fitted policies to NumPyro's leading enumeration and batch dimensions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy import CategoricalPolicy, DecisionInformation


def policy_distribution(
    rule: CategoricalPolicy, observations: DecisionInformation
) -> dist.CategoricalLogits:
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
        batch shape and a final logits axis in the rule's action order.

    Raises:
        TypeError: If observations include an undeclared information parent.
        KeyError: If a declared information parent is missing.
        ValueError: If parent values do not retain their fitted shapes, leading
            dimensions cannot broadcast, or the rule returns invalid logits shapes.
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

    def logits(information: DecisionInformation) -> jax.Array:
        return rule._distribution(information).logits

    evaluate = logits
    for _ in batch_shape:
        evaluate = jax.vmap(evaluate)
    return dist.Categorical(logits=evaluate(values))
