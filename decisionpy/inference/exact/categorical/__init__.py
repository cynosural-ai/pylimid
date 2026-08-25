"""
Exact inference on categorical (all-discrete) diagrams.

Two engines share the discrete factor machinery in
decisionpy.inference.utils: variable elimination for Bayesian-network
marginals (query) and bucket elimination for influence-diagram policies
(solve). Both require every chance and decision node to declare
``states``.
"""

from decisionpy.inference.exact.categorical.bucket_elim import (  # noqa: F401
    Policy,
    Solution,
    solve,
)
from decisionpy.inference.exact.categorical.variable_elim import (  # noqa: F401
    query,
)

__all__ = ["Policy", "Solution", "query", "solve"]
