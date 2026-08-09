"""
Bucket elimination — exact policies for discrete influence diagrams.

The graph-native counterpart to variable elimination: the same factor machinery,
but for the full influence-diagram objective — maximize expected total utility
by choosing, for each decision, the best action given its observed information
set.

Algorithm
---------
1. Build the probability factors (one CPT per chance node) and the utility
   factors (one ``U(parents)`` per utility node). Utility factors combine
   **additively** into a single factor, because the objective is
   ``E[sum_k U_k]`` — unlike probability factors, which multiply.
2. Eliminate variables in reverse topological order (sinks first). For each
   variable *X*:
   - if *X* is a chance node: multiply the probability factors whose scope
     contains *X*. If *X* is in the utility factor's scope, fold the product
     into it (``sum_X P * U``); otherwise sum *X* out of the product and keep
     the result among the probability factors.
   - if *X* is a decision node: multiply any probability factors containing
     it into the utility factor and **max** it out — for every instantiation
     of the decision's information set, record the best action as its policy
     and keep the best value as the utility factor over the remaining
     variables.
3. Multiply what remains: the maximum expected utility.

The reverse-topological order guarantees that when a decision is maximized its
information set (its parents) is still present in the factors, while the chance
variables it influences have already been folded in. A decision is exact for
the structures it accepts: if the best action would have to depend on a
variable outside its information set (a non-regular LIMID), :func:`solve`
raises — that structure needs arc reversal, deferred to a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from decisionpy.graph.chance_node import ChanceNode
from decisionpy.graph.decision_node import DecisionNode
from decisionpy.graph.diagram import Snapshot
from decisionpy.graph.utility_node import UtilityNode
from decisionpy.inference._cpt import cardinalities, cpt, utility_factor
from decisionpy.inference._factor import Factor, _assignments, _project, _union_vars

__all__ = ["Policy", "Solution", "solve"]

#: Optimal policy: decision name -> {information-set assignment: chosen action}.
Policy = dict[str, dict[tuple[int, ...], int]]


@dataclass(frozen=True)
class Solution:
    """
    Result of solving an influence diagram.

    Attributes:
        policy: Per-decision optimal action for every instantiation of the
            decision's information set.
        expected_utility: Maximum expected total utility under *policy*.
    """

    policy: Policy
    expected_utility: float


def solve(snapshot: Snapshot) -> Solution:
    """
    Solve *snapshot* by bucket elimination.

    Requires an all-categorical diagram (every chance and decision node must
    declare ``states``); utility nodes are deterministic and exempt.

    Args:
        snapshot: A validated influence-diagram snapshot.

    Returns:
        A :class:`Solution` with the optimal per-decision policy and the
        maximum expected utility.

    Raises:
        TypeError: If the diagram is not all-categorical.
        NotImplementedError: If a decision's best action would depend on a
            variable outside its information set (a non-regular LIMID —
            requires arc reversal, not yet supported).
    """
    card = cardinalities(snapshot)
    node_map = dict(snapshot.nodes)

    # 1. Probability factors (CPTs) and a single additive utility factor.
    probs: list[Factor] = []
    utilities: list[Factor] = []
    for _name, node in snapshot.nodes:
        if isinstance(node, ChanceNode):
            probs.append(cpt(node, card))
        elif isinstance(node, UtilityNode):
            utilities.append(utility_factor(node, card))
    utility = _sum_utilities(utilities)

    # 2. Eliminate in reverse topological order.
    policy: Policy = {}
    for name in reversed(snapshot.order):
        node = node_map[name]
        if isinstance(node, UtilityNode):
            continue
        relevant = [f for f in probs if name in f.variables]
        if isinstance(node, ChanceNode):
            if not relevant:
                continue  # defensive: every chance node contributes a CPT
            psi = _product(relevant)
            probs = [f for f in probs if f not in relevant]
            if name in utility.variables:
                combined = psi * utility
                utility = combined.marginal(
                    [v for v in combined.variables if v != name]
                )
            else:
                probs.append(
                    psi.marginal([v for v in psi.variables if v != name])
                )
        else:
            assert isinstance(node, DecisionNode)
            if not relevant:
                if name not in utility.variables:
                    continue  # the decision influences nothing: default policy
                base = utility
            else:
                psi = _product(relevant)
                probs = [f for f in probs if f not in relevant]
                base = psi * utility
            policy[name], utility = _max_out(base, node, card)

    _complete_policy(policy, node_map, card)

    # 3. Multiply what remains: the maximum expected utility.
    total = utility
    for f in probs:
        total = total * f
    return Solution(policy=policy, expected_utility=total.values[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sum_utilities(factors: list[Factor]) -> Factor:
    """
    Combine utility factors additively into a single factor.

    Expected total utility is ``E[sum_k U_k]``, so the per-node utility factors
    broadcast-add into one factor over the union of their scopes. The constant
    zero factor represents a diagram with no utility nodes.
    """
    if not factors:
        return Factor(variables=[], card=(), values=[0.0])
    result = factors[0]
    for f in factors[1:]:
        result = _add(result, f)
    return result


def _add(a: Factor, b: Factor) -> Factor:
    """Pointwise sum of two factors, broadcast over the union of scopes."""
    all_vars = _union_vars(a.variables, b.variables)
    all_cards = {
        v: (a.card_of(v) if v in a.variables else b.card_of(v)) for v in all_vars
    }
    lookup = {v: i for i, v in enumerate(all_vars)}
    values = [
        a[_project(assign, a.variables, lookup)]
        + b[_project(assign, b.variables, lookup)]
        for assign in _assignments(all_vars, all_cards)
    ]
    return Factor(
        variables=all_vars,
        card=tuple(all_cards[v] for v in all_vars),
        values=values,
    )


def _product(factors: list[Factor]) -> Factor:
    """Pointwise product of all factors (identity 1 for an empty list)."""
    if not factors:
        return Factor(variables=[], card=(), values=[1.0])
    result = factors[0]
    for f in factors[1:]:
        result = result * f
    return result


def _max_out(
    base: Factor,
    decision: DecisionNode,
    card: dict[str, int],
) -> tuple[dict[tuple[int, ...], int], Factor]:
    """
    Maximize *decision* out of factor *base*, recording its policy.

    The decision's parents are its information set: the best action is chosen
    per instantiation of that information set and returned as the policy
    sub-dict. The best action must not depend on any variable outside the
    information set — if it does, the structure is not regular and cannot be
    solved without arc reversal.

    Args:
        base: A factor whose scope contains the decision.
        decision: The decision node to maximize out.
        card: Variable-name to domain-size map.

    Returns:
        A ``(sub_policy, max_factor)`` pair: the per-information-set action
        choice, and the factor over ``scope(base) - {decision}`` holding the
        best value per context assignment.

    Raises:
        NotImplementedError: If the best action depends on a variable outside
            the decision's information set.
    """
    decision_name = decision.name
    info = list(decision.parents)
    d_idx = base.variables.index(decision_name)
    decision_card = card[decision_name]
    others = [v for v in base.variables if v != decision_name]
    other_cards = [card[v] for v in others]

    # Best (value, action) per context assignment of the other variables.
    best: dict[tuple[int, ...], tuple[float, int]] = {}
    values: list[float] = []
    for ctx in product(*[range(c) for c in other_cards]):
        full = [0] * len(base.variables)
        for i, v in enumerate(others):
            full[base.variables.index(v)] = ctx[i]
        best_value = float("-inf")
        best_action = 0
        for a in range(decision_card):
            full[d_idx] = a
            value = base[tuple(full)]
            if value > best_value:
                best_value, best_action = value, a
        best[ctx] = (best_value, best_action)
        values.append(best_value)

    # Per information-set assignment, the best action must be unambiguous —
    # otherwise the optimal policy would need a variable the decision cannot
    # observe at decision time.
    info_present = [v for v in others if v in info]
    present_positions = [others.index(v) for v in info_present]
    groups: dict[tuple[int, ...], set[int]] = {}
    for ctx, (_value, action) in best.items():
        key = tuple(ctx[p] for p in present_positions)
        groups.setdefault(key, set()).add(action)
    sub_policy: dict[tuple[int, ...], int] = {}
    for info_assign in product(*[range(card[v]) for v in info]):
        key = tuple(info_assign[info.index(v)] for v in info_present)
        actions = groups[key]
        if len(actions) > 1:
            raise NotImplementedError(
                f"Decision {decision_name!r} has different best actions "
                f"for different values of variables outside its information "
                f"set {info!r}; this structure requires arc reversal, which "
                f"is not yet supported. Keep each decision's dependencies "
                f"within its information set."
            )
        sub_policy[info_assign] = next(iter(actions))

    return sub_policy, Factor(
        variables=others, card=tuple(other_cards), values=values
    )


def _complete_policy(
    policy: Policy,
    node_map: dict[str, object],
    card: dict[str, int],
) -> None:
    """
    Fill default policies (action 0) for decisions that were never maxed.

        A decision that influences no utility is skipped by the elimination loop;
        its choice is irrelevant, so any action is optimal. Give it a total policy
        so ``policy`` covers every decision.
        """
    for name, node in node_map.items():
        if isinstance(node, DecisionNode) and name not in policy:
            policy[name] = {}
            for info_assign in product(*[range(card[v]) for v in node.parents]):
                policy[name][info_assign] = 0
