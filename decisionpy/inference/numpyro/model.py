"""
NumPyro model factory — translate a validated snapshot into a NumPyro model.

Internal module.  Public convenience functions that wrap this are in
:mod:`decisionpy.inference.numpyro`.

This is the bridge between the backend-agnostic graph layer and a concrete
probabilistic programming system. It consumes a
:class:`~decisionpy.graph.diagram.Snapshot`
and returns a plain NumPyro model function the caller feeds to ``numpyro.infer``
(e.g. ``Predictive``, ``MCMC``, ``SVI``).

NumPyro is an *optional* dependency: this module is not imported by anything in
:mod:`decisionpy.graph`. :func:`to_model` raises a clear error if NumPyro is not
installed.

Unobserved discrete nodes are detected automatically via ``node.is_discrete``
and annotated with ``infer={"enumerate": "parallel"}`` so that NumPyro's NUTS
and SVI engines can marginalise them out. Observed discrete nodes skip
enumeration (the value is clamped).

Forward / prior-predictive sampling: pass ``observed=None`` (the default) and
feed the model to ``Predictive``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from decisionpy.graph.diagram import Snapshot

__all__ = ["to_model"]


def to_model(
    snapshot: Snapshot,
    observed: dict[str, Any] | None = None,
) -> Callable[[], dict[str, Any]]:
    """
    Build a NumPyro model function from a validated snapshot.

    The returned ``model()`` walks the snapshot's nodes in topological order
    (parents before children — guaranteed by the snapshot), resolves each node's
    parents by name against the upstream sampled values, calls the node's
    ``dist`` factory, and emits one ``numpyro.sample`` site per node.

    Unobserved discrete nodes are auto-detected and annotated with
    ``infer={"enumerate": "parallel"}`` so that NUTS and SVI can marginalise
    them out. Observed nodes skip enumeration and are clamped via ``obs=``.
    Continuous nodes are left for the inference engine.

    :param Snapshot snapshot: A validated, topologically-ordered view of a
        chance-node diagram.
    :param dict | None observed: Mapping of node names to observed values.
        Nodes present here are conditioned on their given value; nodes absent
        are treated as latent. Pass ``None`` or ``{}`` for forward sampling.
    :returns: A NumPyro model function taking no arguments, returning a
        ``dict[str, Any]`` of sampled (or observed) values keyed by node name.
    :raises ImportError: If NumPyro is not installed.
    """
    try:
        import numpyro
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch test
        raise ImportError(
            "The NumPyro backend requires the 'numpyro' extra. "
            "Install it with: pip install decisionpy[numpyro]"
        ) from e

    nodes = snapshot.nodes
    observed = observed or {}

    def model() -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, node in nodes:
            parent_values = {parent: values[parent] for parent in node.parents}
            assert node.dist is not None
            fn = node.dist(**parent_values)
            if name in observed:
                values[name] = numpyro.sample(name, fn, obs=observed[name])
            elif node.is_discrete:
                values[name] = numpyro.sample(name, fn, infer={"enumerate": "parallel"})
            else:
                values[name] = numpyro.sample(name, fn)
        return values

    return model
