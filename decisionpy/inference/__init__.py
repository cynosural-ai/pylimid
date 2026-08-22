"""
Inference layer — engines for Bayesian networks and influence diagrams.

Public API::

    from decisionpy.inference import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    result["rain"].values  # probability vector (Marginal) or draws (Draws)

    solution = solve(diagram)
    solution.policy  # decision name -> info-set assignment -> action
    solution.expected_utility

Engine-specific entry points are public too: ``ve.query`` returns exact
probability vectors, ``id.solve`` returns exact policies, and
``numpyro.samples`` returns raw posterior draws; ``infer`` and ``solve``
auto-dispatch and normalize to typed results.

See :mod:`decisionpy.inference.engine` for the full dispatch logic.
"""

from decisionpy.inference.engine import (  # noqa: F401
    Draws,
    InferenceError,
    InferenceResult,
    Marginal,
    Policy,
    Solution,
    infer,
    solve,
)
