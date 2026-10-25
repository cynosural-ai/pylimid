"""Select and run an influence-diagram solver."""

from __future__ import annotations

from typing import Literal

import jax

from pylimid.errors import InferenceError
from pylimid.graph.diagram import InfluenceDiagram
from pylimid.graph.node import NodeKind
from pylimid.solver.backward_induction import solve as backward_induction_solve
from pylimid.solver.policy_gradient import solve as policy_gradient_solve
from pylimid.solver.regularity import is_solvable
from pylimid.solver.scan.batched import solve as batched_solve
from pylimid.solver.solution import Solution

__all__ = ["SolveMethod", "solve"]

#: Solver selection for solve().
SolveMethod = Literal["auto", "backward_induction", "scan", "policy_gradient"]


def solve(
    diagram: InfluenceDiagram,
    *,
    method: SolveMethod = "auto",
    num_samples: int = 2000,
    num_steps: int | None = None,
    learning_rate: float | None = None,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Find a policy that maximises expected total utility.

    All methods estimate the sum of all utility nodes by Monte Carlo simulation.
    Table solvers require finite decisions with categorical information sets.
    The policy-gradient solver jointly fits finite decisions with numerical,
    categorical, or mixed information parents and declared `LinearSoftmax` families.

    - ``"backward_induction"`` resolves one decision at a time, from the
      last to the first, estimating each action's expected utility per
      information-set assignment by forward sampling. Its cost is additive
      in the decisions. It requires a solvable diagram (see
      `pylimid.solver.is_solvable`), on which it
      finds the optimal policy up to sampling noise.
    - ``"scan"`` enumerates every policy and estimates each one's expected
      utility by forward sampling, keeping the best. It handles any
      diagram with finite decisions and categorical information sets. Its policy count
      grows exponentially with the number of action-table entries.
    - ``"policy_gradient"`` jointly fits all decisions, each declaring
      ``policy_family=LinearSoftmax()``, using sampled-action score gradients and Adam.
      It returns a stochastic `SoftmaxPolicy` per decision and evaluates the joint
      policy on fresh simulations. Each rule uses only its information parents.
      Training has no global or local optimality guarantee, even within the family.
    - ``"auto"`` selects between the two table solvers only: backward induction when
      the diagram is solvable and the scan otherwise. Table solvers reject declared
      policy families; select ``"policy_gradient"`` explicitly to fit one.

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
            for the scan, per action per decision for backward induction, and per
            training update and final evaluation for policy gradients.
        num_steps: Optimizer updates for ``method="policy_gradient"``; defaults
            to 500 for that method. Must be omitted for table solvers.
        learning_rate: Adam step size for ``method="policy_gradient"``; defaults
            to 0.05 for that method. Must be omitted for table solvers.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        A `Solution` containing action tables or fitted stochastic rules, estimated
        expected total utility, and the solver that ran (`Solution.method`, never
        ``"auto"``). Policy-gradient utility is estimated independently of training.

    Raises:
        InferenceError: If the diagram has no decision nodes, if a
            decision's information set lacks named states, if
            ``method="backward_induction"`` is asked for on a diagram that is
            not solvable. Also raised if
            a chance distribution conflicts with its ``states``, or a table solver
            is selected for a decision with a declared policy family.
        ValueError: If the diagram fails `InfluenceDiagram.validate`, *method* is
            unsupported, optimizer settings are invalid or supplied to a table solver,
            any policy-gradient decision lacks a family, or training/evaluation
            produces non-finite results.

    Warns:
        UserWarning: If backward induction never samples an information-set assignment.
            Its action defaults to the first in ``states``. Increasing ``num_samples``
            can help with rare assignments, but impossible assignments cannot be
            sampled.
    """
    if method not in ("auto", "backward_induction", "scan", "policy_gradient"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'auto', "
            "'backward_induction', 'scan' or 'policy_gradient'."
        )

    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    if method == "policy_gradient":
        return policy_gradient_solve(
            snapshot,
            num_samples=num_samples,
            num_steps=500 if num_steps is None else num_steps,
            learning_rate=0.05 if learning_rate is None else learning_rate,
            rng_key=rng_key,
        )
    if num_steps is not None or learning_rate is not None:
        raise ValueError(
            "num_steps and learning_rate require method='policy_gradient'."
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
