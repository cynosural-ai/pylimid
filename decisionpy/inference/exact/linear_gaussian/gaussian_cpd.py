"""
gaussian_cpd — probe a chance node's dist for its linear-Gaussian parameters.

A linear-Gaussian CPD is

    x | parents ~ Normal(loc = a + sum_i b_i * parent_i, scale = sigma)

The dist callable is opaque — the graph layer never inspects it — so the
engine recovers the parameters by evaluating it at chosen parent values:

- a baseline probe (all parents zero; no parents for a root node) yields
  the intercept a and the scale sigma;
- one unit probe per parent yields the slope b_i.

Linearity is verified by reconstructing loc at extra random probe points
and comparing against a + sum_i b_i * x_i; the scale must be constant.
Any CPD that is not a Normal, or whose loc is not affine in its parents,
is rejected loudly — a non-linear loc would silently produce a wrong
Gaussian posterior, so the engine refuses it instead of guessing.
"""

from __future__ import annotations

import numpyro.distributions as dist

from decisionpy.graph.chance_node import ChanceNode

__all__ = ["gaussian_cpd"]

#: Tolerance for the affine-reconstruction check, relative to the scale.
#: Probe arithmetic runs in float32, so rounding noise is ~1e-7; a genuine
#: non-linearity deviates by an order-1 fraction of the scale.
_TOLERANCE = 1e-6

#: Extra probe points per parent, in (-1, 1), used to verify linearity.
_PROBE_POINTS = (-0.7, 0.3, 1.5)


def gaussian_cpd(
    node: ChanceNode,
) -> tuple[float, list[float], float]:
    """
    Probe *node*'s dist for its linear-Gaussian parameters (a, b, scale).

    Args:
        node: A configured continuous chance node whose every parent is
            continuous.

    Returns:
        The intercept a, the per-parent slopes b (in parent order), and
        the conditional standard deviation scale.

    Raises:
        TypeError: If ``dist`` does not return a numpyro Normal.
        ValueError: If the Normal's loc is not affine in the parents, or
            its scale is not constant.
    """
    assert node.dist is not None
    parents = list(node.parents)

    base = node.dist(**dict.fromkeys(parents, 0.0))
    _require_normal(node, base)
    intercept = float(base.loc)
    scale = float(base.scale)

    slopes: list[float] = []
    for parent in parents:
        probe = node.dist(**{p: 1.0 if p == parent else 0.0 for p in parents})
        _require_normal(node, probe)
        slopes.append(float(probe.loc) - intercept)
        _require_constant_scale(node, scale, float(probe.scale))

    # Verify affine reconstruction at extra points (including the scale).
    for point in _PROBE_POINTS:
        for parent in parents:
            probe = node.dist(**{p: point if p == parent else 0.0 for p in parents})
            _require_normal(node, probe)
            expected = intercept + slopes[parents.index(parent)] * point
            if not _close(float(probe.loc), expected, scale):
                raise ValueError(
                    f"Node {node.name!r}'s dist is not linear in its parents: "
                    f"at {parent}={point}, loc={float(probe.loc)} but the "
                    f"affine fit gives {expected}. The linear-Gaussian "
                    f"engine requires every CPD to be Normal with a loc "
                    f"affine in its parents."
                )
            _require_constant_scale(node, scale, float(probe.scale))

    return intercept, slopes, scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_normal(node: ChanceNode, result: object) -> None:
    """Fail loudly unless the probed dist is a numpyro Normal."""
    if not isinstance(result, dist.Normal):
        raise TypeError(
            f"Node {node.name!r}'s dist returned {type(result).__name__}, "
            f"not a Normal; the linear-Gaussian engine requires every "
            f"CPD to be a Normal."
        )


def _require_constant_scale(node: ChanceNode, expected: float, actual: float) -> None:
    """Fail loudly unless the probed scale matches the baseline scale."""
    if not _close(actual, expected, expected):
        raise ValueError(
            f"Node {node.name!r}'s dist has a scale that varies with its "
            f"parents ({expected} at baseline, {actual} elsewhere); the "
            f"linear-Gaussian engine requires a constant scale."
        )


def _close(actual: float, expected: float, reference: float) -> bool:
    """Compare two values to _TOLERANCE, relative to *reference*."""
    return abs(actual - expected) <= _TOLERANCE * max(1.0, abs(reference))
