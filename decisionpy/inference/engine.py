"""
Unified inference entry-point.

::

    from decisionpy.inference import infer

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": [0.456, 0.544]}  (exact posteriors for discrete BNs)
"""

from __future__ import annotations

from decisionpy.graph.diagram import InfluenceDiagram

__all__ = ["InferenceError", "infer"]


class InferenceError(Exception):
    """Asked for an inference method that cannot handle the given diagram."""


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int] | None = None,
    engine: str = "auto",
) -> dict[str, list[float]]:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Args:
        diagram: A validated influence diagram (chance nodes only for v0).
        query: Names of variables whose posteriors are requested.
        observed: Map from node name to its observed integer state.
        engine: ``"auto"`` (default), ``"ve"``, or ``"numpyro"``.

    Returns:
        ``{var_name: [P(0), P(1), ...]}`` — one probability vector per query
        variable. Probabilities sum to 1.

    Raises:
        InferenceError: If the chosen engine cannot handle the diagram.
        ImportError: If ``engine="numpyro"`` and the numpyro extra is not
            installed.
    """
    observed = observed or {}
    snapshot = diagram.snapshot()

    if engine == "auto":
        engine = _choose_engine(snapshot)

    if engine == "ve":
        return _infer_ve(snapshot, query, observed)

    if engine == "numpyro":
        return _infer_numpyro(snapshot, query, observed)

    raise InferenceError(
        f"Unknown engine {engine!r}. Choose 'auto', 've', or 'numpyro'."
    )


# -- engine selection ---------------------------------------------------------


def _choose_engine(snapshot) -> str:
    all_discrete = all(node.is_discrete for _, node in snapshot.nodes)
    if all_discrete:
        return "ve"
    try:
        import numpyro  # noqa: F401
    except ImportError as exc:
        raise InferenceError(
            "Diagram has continuous nodes and the numpyro extra is not "
            "installed. Install with: pip install decisionpy[numpyro]"
        ) from exc
    return "numpyro"


# -- engine backends ----------------------------------------------------------


def _infer_ve(snapshot, query, observed) -> dict[str, list[float]]:
    from decisionpy.inference.ve import query as ve_query

    try:
        return ve_query(snapshot, variables=query, observed=observed)
    except TypeError as e:
        raise InferenceError(
            f"Variable elimination requires an all-discrete diagram. {e}"
        ) from e


def _infer_numpyro(snapshot, query, observed) -> dict[str, list[float]]:
    import jax
    from numpyro.infer import MCMC, NUTS

    from decisionpy.inference.numpyro.model import to_model

    all_discrete = all(node.is_discrete for _, node in snapshot.nodes)
    model = to_model(snapshot, observed=observed)

    if all_discrete:
        return _infer_numpyro_discrete(model, snapshot, query)
    else:
        mcmc = MCMC(
            NUTS(model),
            num_warmup=500,
            num_samples=2000,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0))
        posterior = mcmc.get_samples()
        result = {}
        for name in query:
            if name in posterior:
                result[name] = [float(x) for x in posterior[name]]
            else:
                result[name] = _infer_numpyro_discrete(model, snapshot, [name])[name]
        return result


def _infer_numpyro_discrete(model, snapshot, query) -> dict[str, list[float]]:
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.contrib.funsor import infer_discrete

    num_samples = 2000
    values: dict[str, list[float]] = {}
    for name in query:
        values[name] = []

    for i in range(num_samples):
        k = jax.random.PRNGKey(i)
        inferred = infer_discrete(model, temperature=1, rng_key=k)
        tr = numpyro.handlers.trace(inferred).get_trace()
        for name in query:
            values[name].append(float(tr[name]["value"]))

    result: dict[str, list[float]] = {}
    for name in query:
        node = _find_node(snapshot, name)
        card = len(node.states)
        counts = jnp.bincount(jnp.array(values[name], dtype=jnp.int32), length=card)
        probs = counts / counts.sum()
        result[name] = [float(p) for p in probs]
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
