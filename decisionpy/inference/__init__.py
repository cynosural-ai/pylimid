"""
Inference layer — engines for Bayesian networks and influence diagrams.

Public API::

    from decisionpy.inference import infer

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    result["rain"].values  # probability vector (Marginal) or draws (Draws)

Engine-specific entry points are public too: ``ve.query`` returns exact
probability vectors and ``numpyro.samples`` returns raw posterior draws;
``infer`` auto-dispatches and normalizes to typed per-entry results.

See :mod:`decisionpy.inference.engine` for the full dispatch logic.
"""

from decisionpy.inference.engine import (  # noqa: F401
    Draws,
    InferenceError,
    InferenceResult,
    Marginal,
    infer,
)
