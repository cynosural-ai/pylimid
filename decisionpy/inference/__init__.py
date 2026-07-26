"""
Inference layer — engines for Bayesian networks and influence diagrams.

Public API::

    from decisionpy.inference import infer

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})

See :mod:`decisionpy.inference._api` for the full dispatch logic.
"""

from decisionpy.inference.engine import InferenceError, infer  # noqa: F401
