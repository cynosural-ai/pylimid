"""Posterior inference for chance and utility nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax

from pylimid._numpyro.samplers import samples
from pylimid.errors import InferenceError
from pylimid.graph.diagram import DiagramSnapshot, InfluenceDiagram
from pylimid.graph.nodes import ChanceNode, NodeKind
from pylimid.inference.posterior import InferenceResult, Posterior
from pylimid.policy import CategoricalPolicy, ContinuousPolicy, DecisionRule, Policy

__all__ = ["infer"]


def infer(
    diagram: InfluenceDiagram,
    query: list[str],
    *,
    observed: dict[str, int | float] | None = None,
    policy: Policy | None = None,
    num_samples: int = 10000,
    rng_key: jax.Array | None = None,
) -> InferenceResult:
    """
    Compute posterior marginals for *query* variables given *observed* evidence.

    Only the query and observed nodes and their ancestors take part in
    inference: any other node cannot change the posterior, so it is left out.
    A decision among those ancestors must be bound by *policy*. A fixed
    numerical action clamps it, while a categorical table selects an action from its
    information-set assignment. A deterministic function can select an action from
    continuous or mixed observations during forward evaluation. A fitted
    `CategoricalPolicy` or `ContinuousPolicy` supplies a conditional action
    distribution using learned probabilities or densities, supporting both forward
    evaluation and conditioning on chance
    evidence. Its information parents may be categorical, continuous, or mixed.
    Function policies support forward evaluation only. A decision outside the
    relevant ancestors needs no policy.

    Under fitted policies, evidence downstream of a decision updates beliefs about
    its action and information parents. Fitted parameters stay fixed, and each rule
    uses only its declared information parents. Decisions remain internal variables;
    queries return chance variables and utilities.

    A utility in *query* is computed from its parents in every draw, which
    gives the distribution of the payoff under the bound actions and the
    evidence. Utilities never affect the posterior of other variables.

    With no observed chance node, draws come from forward sampling. With evidence,
    NumPyro uses exact enumeration for categorical chance variables, categorical
    fitted actions, and supported finite counts. NUTS samples remaining continuous
    chance variables and fitted continuous actions. Scalar
    count latents require NumPyro enumeration of a fixed integer range starting at
    zero. For Binomial, use a Python integer ``total_count``; probabilities may depend
    on other variables.
    Unbounded, shifted and batched count latents remain unsupported. NumPyro and JAX
    report errors if enumeration bounds are non-static, including parent-dependent
    trial counts. Counts with only one possible value, zero, are clamped exactly.
    Returned draws and their summaries are Monte Carlo estimates in all cases.

    The whole diagram is validated before selecting the relevant nodes. Chance
    distributions and utility functions must be JAX-traceable. Custom policy
    families must also support JAX transformations and differentiation with respect
    to continuous latent information used in the likelihood for NUTS.

    Args:
        diagram: The influence diagram to query. Validation runs automatically.
        query: Names of chance and utility nodes whose posteriors are requested.
            Chance nodes may have scalar or array-valued draws; utilities are scalar.
        observed: Map from chance node name to its observed value: an integer state
            index for a categorical variable, an actual count for a numerical discrete
            variable, or a real value for a continuous one.
        policy: Map from decision names to fixed numerical actions, contingent action
            tables mapping parent-state tuples to action indices, JAX-traceable
            deterministic functions, or fitted `CategoricalPolicy` or
            `ContinuousPolicy` rules. Functions accept information parents by keyword
            and return a scalar integer index for categorical actions or a real scalar
            for continuous actions. They support forward evaluation only, without
            observed chance nodes. Fitted rules contribute their conditional action
            probabilities or densities to forward sampling and posterior conditioning.
            Numerical information parents retain their fitted scalar or array shapes.
            A table maps each tuple of parent-state indices, in the decision's
            ``parents`` order, to an action index in its ``states``. A decision with
            no parents uses ``()`` as its table key. Every decision that is an
            ancestor of a query or observed node must appear.
        num_samples: Number of posterior draws per query variable. NUTS
            runs 500 warmup steps before them.
        rng_key: JAX PRNG key; defaults to ``jax.random.PRNGKey(0)``, so
            repeated calls return the same estimate.

    Returns:
        One `Posterior` per query variable, keyed by its name. Entries at the same
        position in each posterior's ``values`` belong to the same joint draw. Use
        `Posterior.marginal` for category probabilities, and `Posterior.mean`,
        `Posterior.std` or `Posterior.hdi` for numerical variables and utilities.
        Draws and numerical summaries are JAX arrays. Summaries have the node's shape,
        including zero-dimensional arrays for scalar nodes.

    Raises:
        InferenceError: If a decision that influences the query or the
            evidence is unbound, if *policy* names a node that is not a
            decision or contains an incomplete or invalid action table, if
            *query* names a decision, or if *observed* names a node that is
            not a chance node, or if a numerical count latent lacks supported
            enumeration or has a non-scalar batch shape.
        KeyError: If *query* or *observed* names a node that is not in the
            diagram.
        ValueError: If the diagram fails `InfluenceDiagram.validate`, or a table policy
            has an information-set parent without named states, or a relevant chance
            distribution conflicts with its ``states``, or fitted policy inputs have
            incompatible shapes.
        NotImplementedError: If a relevant function policy uses chance-node evidence,
            or continuous action support metadata is unsupported.
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
    _validate_observed(snapshot, observed)

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
    contingent_policy: dict[str, DecisionRule] = {}
    bound: dict[str, int | float | jax.Array] = {}
    for name, rule in policy.items():
        if name not in relevant:
            continue
        if isinstance(rule, (dict, CategoricalPolicy, ContinuousPolicy)):
            contingent_policy[name] = cast(DecisionRule, rule)
        elif callable(rule):
            contingent_policy[name] = cast(Callable[..., int | float | jax.Array], rule)
        else:
            bound[name] = rule
    draws = samples(
        pruned,
        observed={**observed, **bound},
        query=query,
        num_samples=num_samples,
        rng_key=rng_key,
        policy=contingent_policy,
    )
    nodes = dict(pruned.nodes)
    result: InferenceResult = {}
    for name in query:
        node = nodes[name]
        result[name] = Posterior(
            values=draws[name],
            states=node.states if isinstance(node, ChanceNode) else None,
        )
    return result


def _validate_query(snapshot, query: list[str]) -> None:
    for name in query:
        node = _find_node(snapshot, name)
        if node.kind is NodeKind.DECISION:
            raise InferenceError(
                f"query= names chance and utility nodes only; {name!r} is a "
                "decision node. Bind decision behavior with policy= and query "
                "chance variables or utilities."
            )


def _validate_observed(snapshot, observed: dict[str, int | float]) -> None:
    for name in observed:
        node = _find_node(snapshot, name)
        if node.kind is not NodeKind.CHANCE:
            raise InferenceError(
                f"observed= names chance nodes only; {name!r} is a "
                f"{node.kind.value} node. Bind decisions with policy=; "
                "utilities are computed, not observed."
            )


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


def _find_node(snapshot, name: str):
    for n, node in snapshot.nodes:
        if n == name:
            return node
    raise KeyError(name)
