"""
Exact inference engines, grouped by model class.

Each subpackage handles one model class — categorical, linear-Gaussian,
conditional linear-Gaussian — and owns the preconditions of that class,
failing loudly (InferenceError / TypeError) when a diagram does not
satisfy them. These engines are opt-in: the NumPyro backend is the default
engine, and exact engines are selected explicitly via engine=.
"""

from decisionpy.inference.exact.categorical import (  # noqa: F401
    Policy,
    Solution,
    query,
    solve,
)
