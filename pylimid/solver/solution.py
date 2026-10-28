"""Fitted stochastic policies and independently estimated expected utility."""

from dataclasses import dataclass
from itertools import product

from pylimid.policy import ContinuousPolicy, FittedPolicy, TabularSoftmax

__all__ = ["Solution"]


@dataclass(frozen=True)
class Solution:
    """
    Result of joint policy-gradient optimization.

    Attributes:
        policy: One fitted `CategoricalPolicy` or `ContinuousPolicy` per decision,
            using only its declared information parents. Each rule retains its
            family, parameters, and action domain.
        expected_utility: Independent Monte Carlo estimate of expected total utility
            under the final stochastic policy, with no optimality or convergence
            guarantee.
    """

    policy: FittedPolicy
    expected_utility: float

    def render(self) -> str:
        """
        Render the fitted policies using their stored information and action labels.

        Tabular families show the action probabilities for every information
        assignment. Other families are identified by name, with probabilities shown
        for categorical decisions without parents. Continuous actions show their
        domain and, without parents, the distribution type. Continuous information
        is not enumerated.

        Returns:
            A multi-line string headed by the estimated expected total utility.
        """
        lines = [f"expected utility: {self.expected_utility:.2f}"]
        for name, rule in self.policy.items():
            lines += [
                "",
                f"{name} ({', '.join(rule.parents) or 'no parents'})",
                f"  fitted {type(rule.family).__name__} policy (stochastic)",
            ]
            if isinstance(rule, ContinuousPolicy):
                lines.append(f"  continuous action space: {rule.action_space}")
                if not rule.parents:
                    lines.append(
                        f"  action distribution: {type(rule.distribution()).__name__}"
                    )
            elif isinstance(rule.family, TabularSoftmax):
                labels = []
                for states in rule.parent_states:
                    assert states is not None
                    labels.append(states)
                assignments = product(*(range(len(states)) for states in labels))
                for assignment in assignments:
                    row = rule.probabilities(
                        **dict(zip(rule.parents, assignment, strict=True))
                    ).tolist()
                    situation = (
                        ", ".join(
                            f"{parent}={states[state]}"
                            for parent, states, state in zip(
                                rule.parents, labels, assignment, strict=True
                            )
                        )
                        or "always"
                    )
                    lines.append(_row(situation, rule.states, row))
            elif not rule.parents:
                lines.append(_row("always", rule.states, rule.probabilities().tolist()))
        return "\n".join(lines)


def _row(situation, states, probabilities):
    """Format an information assignment and its labeled action probabilities."""
    actions = ", ".join(
        f"{state}={probability:.3f}"
        for state, probability in zip(states, probabilities, strict=True)
    )
    return f"  {situation} -> {actions}"
