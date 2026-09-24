"""
Influence-diagram solvers — the Monte-Carlo solver family.

Each module implements one approach to solving an influence diagram:

- intervention_scan: the unbatched scan (Strategy B) — enumerates the
  policy space, one forward-sampled evaluation per policy. Global
  optimum of the Monte-Carlo estimate; exponential in the policy space.
- batched_scan: the same algorithm with every policy evaluated in one
  vmapped, jitted pass. Same policy and expected utility as the unbatched
  scan, at a fraction of the runtime.
- backward_induction: resolves the decisions in reverse order from the
  utilities, estimating the per-assignment continuation values by
  stratified forward sampling. Additive in the decisions; requires a
  solvable (regular) diagram.
"""

from decisionpy.inference.numpyro.solvers.backward_induction import (
    solve as backward_induction_solve,
)
from decisionpy.inference.numpyro.solvers.batched_scan import solve as batched_solve
from decisionpy.inference.numpyro.solvers.intervention_scan import solve  # noqa: F401

__all__ = ["backward_induction_solve", "batched_solve", "solve"]
