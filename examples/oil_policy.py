"""
Train stochastic policies on a categorical oil-investment problem.

Run with ``uv run python examples/oil_policy.py``. Utility is measured in
millions of dollars. The chance probabilities and rewards match the oil tutorial.
"""

from __future__ import annotations

import argparse
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from pylimid import (
    ChanceNode,
    DecisionNode,
    InfluenceDiagram,
    Policy,
    TabularPolicy,
    TabularSoftmax,
    UtilityNode,
    infer,
    solve,
)

_PRIOR = np.array([0.35, 0.45, 0.20])
_REWARD = np.array([1250.0, 630.0, 0.0])
_RESULT = np.array(
    [
        [[0.95, 0.05, 0.0], [0.70, 0.30, 0.0], [0.15, 0.85, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    ]
)
_STATES = {
    "quality": ("high", "medium", "low"),
    "test": ("do", "not_do"),
    "result": ("pass", "fail", "no_result"),
    "buy": ("buy", "not_buy"),
}


def build_diagram() -> InfluenceDiagram:
    """
    Build the oil diagram with a trainable categorical policy at each decision.

    Returns:
        A diagram with explicit test memory and an explicit ``no_result`` state.
    """
    prior, reward, result = (jnp.asarray(value) for value in (_PRIOR, _REWARD, _RESULT))
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            "quality", states=_STATES["quality"], dist=lambda: dist.Categorical(prior)
        )
    )
    diagram.add_node(
        DecisionNode(
            "test",
            states=_STATES["test"],
            policy_family=TabularSoftmax(),
        )
    )
    diagram.add_node(
        ChanceNode(
            "result",
            parents=("quality", "test"),
            states=_STATES["result"],
            dist=lambda quality, test: dist.Categorical(result[test, quality]),
        )
    )
    diagram.add_node(
        DecisionNode(
            "buy",
            parents=("test", "result"),
            states=_STATES["buy"],
            policy_family=TabularSoftmax(),
        )
    )
    diagram.add_node(
        UtilityNode(
            "utility",
            parents=("quality", "test", "buy"),
            values=lambda quality, test, buy: (
                jnp.where(buy == 0, reward[quality], 350.0)
                - jnp.where(test == 0, 30.0, 0.0)
            ),
        )
    )
    return diagram


def _probability_tables(policy):
    """Extract the policy's action distributions without sampling or taking argmax."""
    tables = []
    for name, cards in (("test", ()), ("buy", (2, 3))):
        rule = policy[name]
        if isinstance(rule, TabularPolicy):
            scores = np.asarray(rule.logits, dtype=np.float64)
            weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probabilities = weights / weights.sum(axis=-1, keepdims=True)
        else:
            assert isinstance(rule, dict)
            probabilities = np.zeros((*cards, 2))
            for assignment in product(*(range(card) for card in cards)):
                probabilities[(*assignment, rule[assignment])] = 1.0
        tables.append(probabilities)
    return tables


def exact_policy_value(policy: Policy) -> float:
    """
    Evaluate a fixed oil policy by summing all categorical trajectories.

    This includes randomized decisions and action-dependent result probabilities.
    No Monte Carlo estimates or optimizer internals enter the calculation.

    Args:
        policy: Complete deterministic tables or fitted `TabularPolicy` rules for
            ``test`` and ``buy``, using the example's parent and state orders.

    Returns:
        Expected utility of this policy in millions of dollars, up to floating-point
        roundoff, with no sampling error.
    """
    test_probabilities, buy_probabilities = _probability_tables(policy)
    value = 0.0
    for quality, test, result, buy in product(range(3), range(2), range(3), range(2)):
        probability = (
            _PRIOR[quality]
            * test_probabilities[test]
            * _RESULT[test, quality, result]
            * buy_probabilities[test, result, buy]
        )
        reward = _REWARD[quality] if buy == 0 else 350.0
        value += probability * (reward - (30.0 if test == 0 else 0.0))
    return float(value)


def main() -> None:
    """Train the joint policy and compare its value with the analytic oil optimum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-samples", type=int, default=100000)
    args = parser.parse_args()
    diagram = build_diagram()
    solution = solve(
        diagram,
        num_steps=args.steps,
        num_samples=args.samples,
        rng_key=jax.random.PRNGKey(args.seed),
    )
    print(solution.render(diagram))
    exact = exact_policy_value(solution.policy)
    print(f"\nExact value of the stochastic policy: {exact:.6f} million")
    print(f"Loss against the analytic optimum of 721: {721.0 - exact:.6f} million")
    values = np.asarray(
        infer(
            diagram,
            ["utility"],
            policy=solution.policy,
            num_samples=args.evaluation_samples,
            rng_key=jax.random.PRNGKey(1_000_000),
        )["utility"].values,
        dtype=np.float64,
    )
    margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    print(
        f"Independent evaluation: {values.mean():.3f}; approximate 95% interval "
        f"[{values.mean() - margin:.3f}, {values.mean() + margin:.3f}]"
    )
    print("The interval measures evaluation noise, not distance from the optimum.")


if __name__ == "__main__":
    main()
