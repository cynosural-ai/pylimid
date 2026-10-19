"""Select and run an influence-diagram solver."""

from __future__ import annotations

from typing import Literal

import jax

from pylimid.errors import InferenceError
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.node import NodeKind
from pylimid.solver.backward_induction import solve as backward_induction_solve
from pylimid.solver.regularity import is_solvable
from pylimid.solver.scan.batched import solve as batched_solve
from pylimid.solver.solution import Solution

__all__ = ["SolveMethod", "solve"]

#: Solver selection for solve().
SolveMethod = Literal["auto", "backward_induction", "scan"]


def solve(
    diagram: InfluenceDiagram,
    *,
    method: SolveMethod = "auto",
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Find a policy that maximises expected total utility.

    Two Monte Carlo solvers are available, both for discrete decisions with discrete
    information sets. Chance nodes may be discrete, continuous, or both in the same
    diagram. The total utility is the sum of all utility nodes.

    - ``"backward_induction"`` resolves one decision at a time, from the
      last to the first, estimating each action's expected utility per
      information-set assignment by forward sampling. Its cost is additive
      in the decisions. It requires a solvable diagram (see
      `pylimid.solver.is_solvable`), on which it
      finds the optimal policy up to sampling noise.
    - ``"scan"`` enumerates every policy and estimates each one's expected
      utility by forward sampling, keeping the best. It handles any
      diagram with discrete decisions and information sets, but the number of policies
      grows exponentially with the number of action-table entries.
    - ``"auto"`` uses backward induction when the diagram is solvable and
      the scan otherwise.

    Chance distributions and utility functions must be JAX-traceable: use ``jnp``
    operations on parent values, without ``float()`` or ``int()`` conversions or Python
    conditionals on those values.

    With backward induction, parent-state combinations absent from the simulation
    receive the first action in the decision's ``states`` and produce a warning.
    Those policy entries are not optimised; they may represent impossible or rare
    situations.

    Args:
        diagram: An influence diagram with at least one decision. Validation runs
            automatically.
        method: Which solver to run.
        num_samples: Forward samples per evaluation: per candidate policy
            for the scan, per action per decision for backward induction.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        A `Solution` with the optimal per-decision policy (decision name to
        information-set assignment to chosen action), the expected total
        utility (a Monte-Carlo estimate), and the solver that ran
        (`Solution.method`, never ``"auto"``).

    Raises:
        InferenceError: If the diagram has no decision nodes, if a
            decision's information set is not discrete, if
            ``method="backward_induction"`` is asked for on a diagram that is
            not solvable. Also raised if
            a chance distribution conflicts with its ``states``.
        ValueError: If the diagram fails `InfluenceDiagram.validate`, or *method* is not
            one of the supported names.

    Warns:
        UserWarning: If backward induction never samples an information-set assignment.
            Its action defaults to the first in ``states``. Increasing ``num_samples``
            can help with rare assignments, but impossible assignments cannot be
            sampled.
    """
    if method not in ("auto", "backward_induction", "scan"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'auto', "
            "'backward_induction' or 'scan'."
        )

    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    if method == "auto":
        method = "backward_induction" if is_solvable(snapshot) else "scan"
    solver = (
        backward_induction_solve if method == "backward_induction" else batched_solve
    )

    try:
        return solver(snapshot, num_samples=num_samples, rng_key=rng_key)
    except ValueError as e:
        raise InferenceError(f"solve(method={method!r}) failed. {e}") from e
