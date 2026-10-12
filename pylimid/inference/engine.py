"""
Unified inference entry-point.

::

    from pylimid import infer, solve

    result = infer(diagram, query=["rain"], observed={"wet_grass": 1})
    # → {"rain": Posterior(values=[...], states=("no", "yes"))}

    solution = solve(diagram)
    # → Solution(policy={"treat": {(0,): 0, (1,): 1}}, expected_utility=78.3,
    #            method="backward_induction")
"""

from __future__ import annotations

from typing import Literal

import jax

from pylimid.graph.diagram import DiagramSnapshot, InfluenceDiagram
from pylimid.graph.node import NodeKind
from pylimid.inference.numpyro import samples as numpyro_samples
from pylimid.inference.numpyro.solvers import (
    backward_induction_solve,
    batched_solve,
    is_solvable,
)
from pylimid.inference.posterior import InferenceResult, Posterior
from pylimid.inference.solution import Policy, Solution, SolverName

__all__ = [
    "InferenceError",
    "InferenceResult",
    "Policy",
    "Posterior",
    "Solution",
    "SolveMethod",
    "SolverName",
    "infer",
    "solve",
]

#: Solver selection for solve().
SolveMethod = Literal["auto", "backward_induction", "scan"]


class InferenceError(Exception):
    """Asked for an inference method that cannot handle the given diagram."""


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int | float] | None = None,
    policy: dict[str, int] | None = None,
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Only the query and observed nodes and their ancestors take part in
    inference: any other node cannot change the posterior, so it is left out.
    Utilities are always ignored. A decision among those ancestors must be
    bound by *policy*: it is clamped to its chosen action and behaves like
    observed evidence, so the diagram collapses to a Bayesian network. A
    decision outside them needs no policy, and is ignored if it has one.

    The NumPyro engine is the only engine. With no observed chance node the
    draws come from forward sampling; otherwise from exact discrete
    enumeration when every latent variable is discrete, and NUTS when some
    latent variable is continuous. Every result is a Monte-Carlo estimate.

    Args:
        diagram: A validated influence diagram.
        query: Names of chance variables whose posteriors are requested.
        observed: Map from node name to its observed value — an integer
            state for a discrete variable, a real value for a continuous
            one.
        policy: Map from decision name to its chosen integer action. Every
            decision that is an ancestor of a query or observed node must
            appear.
        num_samples: Number of posterior draws per query variable. NUTS
            runs 500 warmup steps before them.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        ``{var_name: result}`` — one `Posterior` per query variable: raw
        posterior draws plus the variable's states (``None`` for
        continuous). Each summary method is valid for exactly one kind:
        `Posterior.marginal` bincounts a discrete posterior into its
        probability vector, and `Posterior.mean` / `Posterior.std` /
        `Posterior.hdi` summarize the draws of a continuous one —
        calling the wrong kind raises. Everything is a Monte-Carlo
        estimate.

    Raises:
        InferenceError: If a decision that influences the query or the
            evidence is unbound, if *policy* names a node that is not a
            decision, or if *query* names a non-chance node.
        KeyError: If *query* or *observed* names a node that is not in the
            diagram.
    """
    observed = observed or {}
    policy = policy or {}
    snapshot = diagram.snapshot()

    decisions = [
        name for name, node in snapshot.nodes if node.kind is NodeKind.DECISION
    ]
    unknown = sorted(set(policy) - set(decisions))
    if unknown:
        raise InferenceError(
            f"policy= binds decision nodes only; unknown names: {unknown}."
        )

    _validate_query(snapshot, query)

    relevant = _ancestors(snapshot, [*query, *observed])
    unbound = [name for name in decisions if name in relevant and name not in policy]
    if unbound:
        raise InferenceError(
            f"Diagram has unbound decision nodes: {unbound}. They influence "
            "the query or the observed nodes, so specify policy={...} for them."
        )

    pruned = DiagramSnapshot(
        nodes=tuple((name, node) for name, node in snapshot.nodes if name in relevant)
    )
    bound = {name: action for name, action in policy.items() if name in relevant}
    return _infer_numpyro(
        pruned, query, {**observed, **bound}, num_samples=num_samples, rng_key=rng_key
    )


def solve(
    diagram: InfluenceDiagram,
    *,
    method: SolveMethod = "auto",
    num_samples: int = 2000,
    rng_key: jax.Array | None = None,
) -> Solution:
    """
    Solve an influence diagram: the optimal policy and its expected utility.

    Two Monte-Carlo solvers are available, both for discrete decisions with
    discrete information sets; chance nodes may be discrete, continuous or
    mixed.

    - ``"backward_induction"`` resolves one decision at a time, from the
      last to the first, estimating each action's expected utility per
      information-set assignment by forward sampling. Its cost is additive
      in the decisions. It requires a solvable diagram (see
      `pylimid.inference.numpyro.solvers.is_solvable`), on which it
      finds the optimal policy up to sampling noise.
    - ``"scan"`` enumerates every policy and estimates each one's expected
      utility by forward sampling, keeping the best. It handles any
      diagram, but its cost grows exponentially with the policy space.
    - ``"auto"`` uses backward induction when the diagram is solvable and
      the scan otherwise.

    Utility ``values`` callables must be JAX-traceable (pure ``jnp``
    expressions, no ``float()`` / ``int()`` coercion): both solvers
    evaluate them inside a vectorized forward pass.

    Args:
        diagram: A validated influence diagram with at least one decision.
        method: Which solver to run.
        num_samples: Forward samples per evaluation: per candidate policy
            for the scan, per action per decision for backward induction.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        A `Solution` with the optimal per-decision policy (decision name to
        information-set assignment to chosen action), the expected total
        utility (a Monte-Carlo estimate), and the solver that ran
        (`Solution.method`, never ``"auto"``).

    Raises:
        InferenceError: If the diagram has no decision nodes, if a
            decision's information set is not discrete, if
            ``method="backward_induction"`` is asked for on a diagram that is
            not solvable, or if an information-set assignment is never
            sampled (increase *num_samples*).
        ValueError: If *method* is not one of the supported names.
    """
    if method not in ("auto", "backward_induction", "scan"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'auto', "
            "'backward_induction' or 'scan'."
        )

    snapshot = diagram.snapshot()

    if not any(node.kind is NodeKind.DECISION for _, node in snapshot.nodes):
        raise InferenceError(
            "solve() requires an influence diagram with at least one decision "
            "node; this diagram has none. Use infer() for Bayesian networks."
        )

    if method == "auto":
        method = "backward_induction" if is_solvable(snapshot) else "scan"
    solver = (
        backward_induction_solve if method == "backward_induction" else batched_solve
    )

    try:
        return solver(snapshot, num_samples=num_samples, rng_key=rng_key)
    except ValueError as e:
        raise InferenceError(f"solve(method={method!r}) failed. {e}") from e


def _validate_query(snapshot, query: list[str]) -> None:
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is not NodeKind.CHANCE:
            raise InferenceError(
                f"query= names chance variables only; {name!r} is a "
                f"{node.kind.value} node. Decisions are clamped by policy= "
                f"and utilities are not random variables."
            )


# -- backend ------------------------------------------------------------------


def _ancestors(snapshot: DiagramSnapshot, names: list[str]) -> set[str]:
    """*names* together with every node upstream of them."""
    parents = {name: node.parents for name, node in snapshot.nodes}
    found: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(parents[name])
    return found


def _infer_numpyro(
    snapshot, query, observed, *, num_samples: int, rng_key: jax.Array | None
) -> InferenceResult:
    # ``samples()`` returns raw draws; ``infer()`` wraps them in a
    # Posterior, keeping the draws and the variable's states together.
    # Everything is a Monte-Carlo estimate.
    draws = numpyro_samples(
        snapshot,
        observed=observed,
        query=query,
        num_samples=num_samples,
        rng_key=rng_key,
    )
    result: InferenceResult = {}
    for name in query:
        node = _find_node(snapshot, name)
        result[name] = Posterior(
            values=[float(x) for x in draws[name]], states=node.states
        )
    return result


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
