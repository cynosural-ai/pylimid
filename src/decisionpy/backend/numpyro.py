"""
NumPyro backend — translate a validated snapshot into a NumPyro model.

This is the bridge between the backend-agnostic graph layer and a concrete
probabilistic programming system. It consumes a :class:`~decisionpy.graph.diagram.Snapshot`
— the validated, topologically-ordered view produced by
:meth:`~decisionpy.graph.diagram.InfluenceDiagram.snapshot` — and returns a plain
NumPyro model function the caller feeds to ``numpyro.infer`` (e.g.
``Predictive``, ``MCMC``, ``SVI``).

NumPyro is an *optional* dependency: this module is not imported by anything in
:mod:`decisionpy.graph`, and importing :mod:`decisionpy` or
:mod:`decisionpy.backend` never pulls it in. :func:`to_model` raises a clear
error if NumPyro is not installed. Install the extra with
``pip install decisionpy[numpyro]``.

v0 scope
--------
Chance nodes only — i.e. a Bayesian network. Each node is emitted as::

    fn = node.dist(**resolved_parents)
    value = numpyro.sample(name, fn)  # unobserved, continuous
    value = numpyro.sample(name, fn, obs=data[name])  # observed
    value = numpyro.sample(name, fn, infer={"enumerate": "parallel"})
    # unobserved, discrete — marginalised analytically during inference

Unobserved discrete nodes are detected automatically via ``node.is_discrete``
and annotated with ``infer={"enumerate": "parallel"}`` so that NumPyro's NUTS
and SVI engines can marginalise them out. Observed discrete nodes skip
enumeration (the value is clamped).

Forward / prior-predictive sampling continues to work: pass ``observed=None``
(the default) and feed the model to ``Predictive``.

with parents resolved by *name* (see the contract in
[`diagram.md`](../../docs/diagram.md) and the ``dist`` calling convention in
:mod:`decisionpy.graph.chance_node`). This layer does translation only; how the
returned model is then sampled or inferred is the caller's concern.

Open seam, deliberately not solved in v0
----------------------------------------
A :class:`Snapshot` holds references to the still-mutable nodes. The node
references are captured once, at :func:`to_model` time, so later structural
edits to the live diagram (adding/removing nodes or edges) do not affect an
already-captured model. But field mutation on a captured node *after*
:func:`to_model` but before sampling (e.g. reassigning ``node.dist``) would be
seen by the model. A genuinely deep-frozen snapshot is the documented future
tightening (see :class:`~decisionpy.graph.diagram.Snapshot`); v0 assumes the
caller does not mutate captured nodes.
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

    Usage::

        from numpyro.infer import MCMC, NUTS, Predictive

        model = to_model(snapshot)
        prior = Predictive(model, num_samples=1000)(rng_key)

        model = to_model(snapshot, observed={"wet_grass": jnp.array(1)})
        mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=2000)
        mcmc.run(rng_key)
        posterior = mcmc.get_samples()
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
            fn = node.dist(**parent_values)
            if name in observed:
                values[name] = numpyro.sample(name, fn, obs=observed[name])
            elif node.is_discrete:
                values[name] = numpyro.sample(name, fn, infer={"enumerate": "parallel"})
            else:
                values[name] = numpyro.sample(name, fn)
        return values

    return model
