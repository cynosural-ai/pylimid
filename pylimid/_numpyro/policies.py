"""Adapt fitted policies to NumPyro's leading enumeration and batch dimensions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy import CategoricalPolicy, ContinuousPolicy, DecisionInformation


def policy_distribution(
    rule: CategoricalPolicy | ContinuousPolicy, observations: DecisionInformation
) -> dist.Distribution:
    """
    Construct an action distribution over broadcast information assignments.

    Each parent's trailing dimensions retain its fitted value shape. Leading
    dimensions broadcast across parents and index separate information assignments;
    each assignment is evaluated through the rule's single-assignment interface.

    Args:
        rule: Fitted stochastic policy with parent value shapes.
        observations: Information-parent values, optionally with leading enumeration
            or batch dimensions.

    Returns:
        A scalar-event action distribution with broadcast leading batch dimensions.
        Categorical probabilities retain their declared action order. Continuous
        distributions retain their densities and transforms.

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
    # vmap preserves scalar shape metadata, including on nested distributions.
    return _rebuild_distribution(evaluate(values))


def _rebuild_distribution(distribution: dist.Distribution) -> dist.Distribution:
    """Rebuild batched constructor parameters and nested scalar distributions."""
    if isinstance(distribution, dist.TransformedDistribution):
        return dist.TransformedDistribution(
            _rebuild_distribution(distribution.base_dist), distribution.transforms
        )
    if isinstance(
        distribution,
        (
            dist.LeftTruncatedDistribution,
            dist.RightTruncatedDistribution,
            dist.TwoSidedTruncatedDistribution,
        ),
    ):
        # vmap repeats its scalar bounds alongside the parameters,
        # but NUTS needs the original scalar support for each action.
        bounds = {
            name: jnp.ravel(value)[0] for name, value in distribution.get_args().items()
        }
        return type(distribution)(
            _rebuild_distribution(distribution.base_dist), **bounds
        )
    return type(distribution)(**distribution.get_args())
