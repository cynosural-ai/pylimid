"""
Influence-diagram solvers — the Monte-Carlo solver family.

decisionpy.solve is the front door; it picks between these solvers. Each
module implements one approach:

- intervention_scan: enumerates the policy space, one forward-sampled
  evaluation per policy. Global optimum of the Monte-Carlo estimate;
  exponential in the policy space. The reference implementation.
- batched_scan: the same algorithm with every policy evaluated in one
  vmapped, jitted pass. Same policy and expected utility as the unbatched
  scan, at a fraction of the runtime.
- backward_induction: resolves the decisions in reverse order from the
  utilities, estimating the per-assignment continuation values by
  stratified forward sampling. Additive in the decisions; requires a
  solvable diagram (see regularity).
- regularity: the solvability check that gates backward induction.
"""

from decisionpy.inference.numpyro.solvers.backward_induction import (
    solve as backward_induction_solve,
)
from decisionpy.inference.numpyro.solvers.batched_scan import solve as batched_solve
from decisionpy.inference.numpyro.solvers.intervention_scan import (
    solve as scan_solve,
)
from decisionpy.inference.numpyro.solvers.regularity import (
    is_solvable,
    solvability_order,
)

__all__ = [
    "backward_induction_solve",
    "batched_solve",
    "is_solvable",
    "scan_solve",
    "solvability_order",
]
