"""
Optimal policy and expected utility — the result of `solve`.

Returned by `solve`. Expected utility is a Monte Carlo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, TypeAlias

import jax

from pylimid.graph.diagram import InfluenceDiagram
from pylimid.policy import Policy
from pylimid.policy_family import SoftmaxPolicy, TabularPolicy

__all__ = ["Solution", "SolverName"]


#: The solver that produced a `Solution`.
SolverName: TypeAlias = Literal["backward_induction", "scan", "policy_gradient"]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: One rule per decision. Table solvers return action tables;
            policy-gradient training returns a fitted stochastic `SoftmaxPolicy` or
            `TabularPolicy` per decision, jointly trained using only each rule's
            information parents.
            Supplied deterministic functions can also be evaluated and rendered.
            Table keys are parent-state tuples in the decision's ``parents`` order;
            values are action indices in its ``states``. A root table uses ``()``.
        expected_utility: Monte-Carlo estimate of the expected total
            utility under *policy*.
        method: The solver that produced the solution. ``"scan"`` evaluated
            every policy, so *policy* is the best of the Monte-Carlo
            estimates; ``"backward_induction"`` resolved one decision at a
            time on a solvable diagram, optimal up to sampling noise.
            ``"policy_gradient"`` fitted a declared family with no optimality
            guarantee, then estimated utility on independent simulations. With
            ``solve(method="auto")`` this records which table solver was chosen.
    """

    policy: Policy
    expected_utility: float
    method: SolverName

    def render(self, diagram: InfluenceDiagram) -> str:
        """
        Render the policy with state labels instead of indices.

        The first line shows the estimated expected utility and the solver used. It is
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
        lines = [
            f"expected utility: {self.expected_utility:.2f} (method: {self.method})"
        ]
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
