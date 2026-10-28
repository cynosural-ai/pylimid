"""Shared information binding and feature encoding for local decision rules."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from pylimid.policy.family import DecisionInformation


def bind_information(
    parents: tuple[str, ...],
    parent_shapes: tuple[tuple[int, ...], ...],
    observations: DecisionInformation,
) -> dict[str, jax.Array]:
    """Order one information assignment and check its declared parent shapes."""
    unexpected = observations.keys() - parents
    if unexpected:
        raise TypeError(f"Unexpected information parents: {sorted(unexpected)}.")
    for parent, shape in zip(parents, parent_shapes, strict=True):
        if jnp.shape(observations[parent]) != shape:
            raise ValueError(f"Information parent {parent!r} must have shape {shape}.")
    return {parent: observations[parent] for parent in parents}


def encode_features(
    observations: DecisionInformation,
    parent_states: Mapping[str, tuple[str, ...] | None],
) -> jax.Array:
    """Encode numerical values and named categories in the information-parent order."""
    features = []
    for parent, value in observations.items():
        states = parent_states[parent]
        value = jnp.asarray(value)
        feature = value if states is None else jax.nn.one_hot(value, len(states))
        features.append(jnp.ravel(feature))
    return jnp.concatenate(features) if features else jnp.empty((0,))
