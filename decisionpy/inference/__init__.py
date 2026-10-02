"""
Inference layer — posterior inference and influence-diagram solving.

Public API::

    from decisionpy import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    result["rain"].values  # raw posterior draws; marginal() for probabilities

    solution = solve(diagram)
    solution.policy  # decision name -> info-set assignment -> action
    solution.expected_utility

The NumPyro engine is the only engine. ``numpyro.samples`` returns raw
posterior draws, and ``numpyro.solvers`` holds the individual solvers that
solve() chooses between; infer() and solve() are the normalized,
ergonomic entry points.
"""

from decisionpy.inference.engine import (
    InferenceError,
    InferenceResult,
    Policy,
    Posterior,
    Solution,
    SolveMethod,
    infer,
    solve,
)

__all__ = [
    "InferenceError",
    "InferenceResult",
    "Policy",
    "Posterior",
    "Solution",
    "SolveMethod",
    "infer",
    "solve",
]
