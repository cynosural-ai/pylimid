"""
Inference layer — posterior inference and influence-diagram solving.

Public API::

    from decisionpy.inference import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    result["rain"].values  # raw posterior draws; marginal() for probabilities

    solution = solve(diagram)
    solution.policy  # decision name -> info-set assignment -> action
    solution.expected_utility

The NumPyro engine is the only engine. The engine-specific entry points
are public too: ``numpyro.samples`` returns raw posterior draws and
``numpyro.solve`` runs the intervention-scan solver directly;
``infer`` and ``solve`` are the normalized, ergonomic entry points.

See decisionpy.inference.engine for the full dispatch logic.
"""

from decisionpy.inference.engine import (  # noqa: F401
    InferenceError,
    InferenceResult,
    Policy,
    Posterior,
    Solution,
    infer,
    solve,
)
