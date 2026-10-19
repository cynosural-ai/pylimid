"""
Optimal policy and expected utility — the result of `solve`.

Returned by `solve`. Expected utility is a Monte Carlo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pylimid.graph.diagram import InfluenceDiagram
from pylimid.policy import Policy

__all__ = ["Solution", "SolverName"]


#: The solver that produced a `Solution`.
SolverName: TypeAlias = Literal["backward_induction", "scan"]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: One action table per decision. Keys are tuples of parent-state indices
            in the decision's ``parents`` order; values are action indices in its
            ``states``. A decision with no parents uses the key ``()``. Actions maximise
            estimated expected utility.
        expected_utility: Monte-Carlo estimate of the expected total
            utility under *policy*.
        method: The solver that produced the solution. ``"scan"`` evaluated
            every policy, so *policy* is the best of the Monte-Carlo
            estimates; ``"backward_induction"`` resolved one decision at a
            time on a solvable diagram, optimal up to sampling noise. With
            ``solve(method="auto")`` this records which one was chosen.
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
        situation ``always``.

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
