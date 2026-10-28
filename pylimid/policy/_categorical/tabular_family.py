"""Trainable tabular softmax family for categorical decisions."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.policy.family import DecisionInformation, PolicyFamily, PolicyParameters


@dataclass(frozen=True)
class TabularSoftmax(PolicyFamily):
    """
    Independent action scores for every scalar categorical information assignment.

    Every parent must declare named finite states and return a scalar state index.
    A parentless decision has one score row. Rows express interactions between
    categorical parents, with no parameter sharing. Their count grows as the product
    of parent cardinalities; rarely visited rows may receive little training.
    """

    def initialize(
        self,
        rng_key: jax.Array,
        *,
        num_actions: int | None,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> PolicyParameters:
        """
        Initialize uniform action probabilities for every information assignment.

        Args:
            rng_key: JAX random key; this family initializes deterministically.
            num_actions: Number of available actions.
            observations: Sample scalar parent-state indices in declared parent order.
            parent_states: Named finite states for every information parent.

        Returns:
            A dictionary containing ``logits`` with one leading axis per parent
            and a final action axis.

        Raises:
            ValueError: If the decision has a continuous action space, or a parent
                lacks named states or has a non-scalar value.
        """
        if num_actions is None:
            raise ValueError("TabularSoftmax requires categorical action states.")
        cards = []
        for parent, value in observations.items():
            labels = parent_states[parent]
            if labels is None:
                raise ValueError(
                    "TabularSoftmax requires named finite states "
                    f"for parent {parent!r}."
                )
            if jnp.shape(value) != ():
                raise ValueError(
                    "TabularSoftmax requires a scalar state index "
                    f"for parent {parent!r}."
                )
            cards.append(len(labels))
        return {"logits": jnp.zeros((*cards, num_actions))}

    def distribution(
        self,
        parameters: PolicyParameters,
        observations: DecisionInformation,
        parent_states: Mapping[str, tuple[str, ...] | None],
    ) -> dist.CategoricalLogits:
        """
        Construct categorical probabilities for one information assignment.

        Args:
            parameters: The current categorical score table under ``logits``.
            observations: Scalar state indices in declared parent order.
            parent_states: Named finite states for every information parent.

        Returns:
            A scalar categorical distribution using the selected score row.
        """
        return dist.Categorical(
            logits=parameters["logits"][tuple(observations.values())]
        )
