"""
Fitted policy and expected utility — the result of `solve`.

Returned by `solve`. Expected utility is a Monte Carlo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax

from pylimid.graph.diagram import InfluenceDiagram
from pylimid.policy import Policy
from pylimid.policy_family import SoftmaxPolicy, TabularPolicy

__all__ = ["Solution"]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: One fitted stochastic rule per decision, jointly trained using
            only its declared information parents. Supplied deterministic tables
            and functions can also be evaluated and rendered.
        expected_utility: Independent Monte Carlo estimate of expected total utility
            under the final stochastic policy. This is an approximate result with
            no optimality or convergence guarantee.
    """

    policy: Policy
    expected_utility: float

    def render(self, diagram: InfluenceDiagram) -> str:
        """
        Render the policy with state labels instead of indices.

        The first line shows the estimated expected utility. It is
        followed by one block per decision: a header naming the decision and its parents
        in information-set order, then one line per situation reading the
        chosen action's label. A decision with no parents has the single
        situation ``always``. Fitted stochastic tables show action probabilities per
        assignment. Function and fitted linear rules are identified without enumerating
        their observation space.

        Args:
            diagram: The diagram this solution was computed from; supplies
                the state labels.

        Returns:
            A multi-line string, for ``print`` or notebook display.
        """
        lines = [f"expected utility: {self.expected_utility:.2f}"]
        for name, rule in self.policy.items():
            decision = diagram[name]
            if isinstance(rule, SoftmaxPolicy):
                lines += [
                    "",
                    f"{name} ({', '.join(rule.parents) or 'no parents'})",
                    "  fitted linear-softmax policy (stochastic)",
                ]
                continue
            if isinstance(rule, TabularPolicy):
                lines += [
                    "",
                    f"{name} ({', '.join(rule.parents) or 'no parents'})",
                    "  fitted tabular-softmax policy (stochastic)",
                ]
                assignments = product(
                    *(range(len(states)) for states in rule.parent_states)
                )
                probabilities = (
                    jax.nn.softmax(rule.logits, axis=-1)
                    .reshape(-1, len(rule.states))
                    .tolist()
                )
                for assignment, row in zip(assignments, probabilities, strict=True):
                    situation = (
                        ", ".join(
                            f"{parent}={states[state]}"
                            for parent, states, state in zip(
                                rule.parents,
                                rule.parent_states,
                                assignment,
                                strict=True,
                            )
                        )
                        or "always"
                    )
                    actions = ", ".join(
                        f"{state}={probability:.3f}"
                        for state, probability in zip(rule.states, row, strict=True)
                    )
                    lines.append(f"  {situation} -> {actions}")
                continue
            if callable(rule):
                lines += [
                    "",
                    f"{name} ({', '.join(decision.parents) or 'no parents'})",
                    "  deterministic function policy",
                ]
                continue
            rows = [
                ", ".join(
                    f"{parent}={diagram.states_of(parent)[state]}"
                    for parent, state in zip(decision.parents, key, strict=True)
                )
                or "always"  # for when the decision node has no parents
                for key in rule
            ]
            width = max(len(row) for row in rows)
            lines += ["", f"{name} ({', '.join(decision.parents) or 'no parents'})"]
            for row, action in zip(rows, rule.values(), strict=True):
                lines.append(f"  {row:<{width}} -> {diagram.states_of(name)[action]}")
        return "\n".join(lines)
