"""
Policy optimisation, solver selection, and solution results.

`solve` chooses backward induction when the diagram is structurally solvable and
the batched policy scan otherwise. The scan implementations and their shared
enumeration helpers live in ``pylimid.solver.scan``. Structural solvability checks
and backward induction live directly in ``pylimid.solver``.
"""

from pylimid.solver.api import SolveMethod, solve
from pylimid.solver.backward_induction import (
    solve as backward_induction_solve,
)
from pylimid.solver.regularity import (
    is_solvable,
    solvability_order,
)
from pylimid.solver.scan.batched import solve as batched_solve
from pylimid.solver.scan.intervention import (
    solve as scan_solve,
)
from pylimid.solver.solution import Solution, SolverName

__all__ = [
    "Solution",
    "SolveMethod",
    "SolverName",
    "solve",
    "backward_induction_solve",
    "batched_solve",
    "is_solvable",
    "scan_solve",
    "solvability_order",
]
