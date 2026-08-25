"""
Exact inference on linear-Gaussian (all-continuous) diagrams.

A linear-Gaussian BN is an all-continuous network whose every node is
Gaussian with a mean linear in its continuous parents. Query() runs
variable elimination over canonical-form Gaussian factors and returns
exact Gaussian posteriors.
"""

from decisionpy.inference.exact.linear_gaussian.gaussian_cpd import (  # noqa: F401
    gaussian_cpd,
)
from decisionpy.inference.exact.linear_gaussian.gaussian_factor import (  # noqa: F401
    GaussianFactor,
)

__all__ = ["GaussianFactor", "gaussian_cpd"]
