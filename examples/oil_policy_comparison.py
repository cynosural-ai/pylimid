"""
Compare the categorical oil problem with exact and sampled policy solvers.

Run with ``uv run python examples/oil_policy_comparison.py``. Utility is measured in
millions of dollars. The chance probabilities and rewards match the oil tutorial.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pyagrum as gum

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


def build_pylimid(*, trainable: bool = False) -> InfluenceDiagram:
    """
    Build the oil diagram with deterministic-table or trainable-table decisions.

    Args:
        trainable: Declare `TabularSoftmax` on both decisions for policy gradients.

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
            policy_family=TabularSoftmax() if trainable else None,
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
            policy_family=TabularSoftmax() if trainable else None,
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


def build_pyagrum() -> gum.InfluenceDiagram:  # ty: ignore[possibly-missing-attribute]
    """
    Build the oil diagram in pyAgrum with the same probabilities and utilities.

    Returns:
        A categorical influence diagram with explicit test memory and no-result state.
    """
    diagram = gum.InfluenceDiagram()  # ty: ignore[possibly-missing-attribute]
    for name, labels in _STATES.items():
        variable = gum.LabelizedVariable(name, name, list(labels))
        if name in ("test", "buy"):
            diagram.addDecisionNode(variable)
        else:
            diagram.addChanceNode(variable)
    diagram.addUtilityNode(gum.LabelizedVariable("utility", "utility", 1))
    for parent, child in (
        ("quality", "result"),
        ("test", "result"),
        ("test", "buy"),
        ("result", "buy"),
        ("quality", "utility"),
        ("test", "utility"),
        ("buy", "utility"),
    ):
        diagram.addArc(parent, child)
    diagram.cpt("quality").fillWith(_PRIOR.tolist())
    for test, quality in product(range(2), range(3)):
        diagram.cpt("result")[{"test": test, "quality": quality}] = _RESULT[
            test, quality
        ].tolist()
        for buy in range(2):
            reward = _REWARD[quality] if buy == 0 else 350.0
            diagram.utility("utility")[
                {"test": test, "quality": quality, "buy": buy}
            ] = reward - (30.0 if test == 0 else 0.0)
    return diagram


def solve_pyagrum(
    diagram: gum.InfluenceDiagram | None = None,  # ty: ignore[possibly-missing-attribute]
) -> tuple[float, Policy]:
    """
    Solve the identical categorical model with pyAgrum's exact LIMID inference.

    Args:
        diagram: A prebuilt oil diagram; omitted to construct the example model.

    Returns:
        Maximum expected utility and a deterministic policy using PyLIMID's indices.
    """
    if diagram is None:
        diagram = build_pyagrum()
    engine = gum.ShaferShenoyLIMIDInference(diagram)  # ty: ignore[possibly-missing-attribute]
    engine.makeInference()
    policy: Policy = {}
    for name, parents, cards in (("test", (), ()), ("buy", ("test", "result"), (2, 3))):
        optimal = engine.optimalDecision(name)
        table = {}
        for assignment in product(*(range(card) for card in cards)):
            information = dict(zip(parents, assignment, strict=True))
            table[assignment] = int(
                np.argmax(
                    [
                        float(optimal[{**information, name: action}])
                        for action in range(2)
                    ]
                )
            )
        policy[name] = table
    return float(engine.MEU()["mean"]), policy


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


def exact_optimum() -> float:
    """
    Enumerate all deterministic oil policies as an independent optimum reference.

    Returns:
        The largest exactly evaluated expected utility among all 128 joint policies.
    """
    assignments = list(product(range(2), range(3)))
    return max(
        exact_policy_value(
            {"test": {(): test}, "buy": dict(zip(assignments, actions, strict=True))}
        )
        for test in range(2)
        for actions in product(range(2), repeat=len(assignments))
    )


def deterministic_policy(policy: Policy) -> Policy:
    """
    Construct the separate policy choosing each row's largest action probability.

    Args:
        policy: Complete oil policies in the example's parent and state orders.

    Returns:
        Deterministic tables requiring their own utility evaluation.
    """
    test_probabilities, buy_probabilities = _probability_tables(policy)
    return {
        "test": {(): int(np.argmax(test_probabilities))},
        "buy": {
            assignment: int(np.argmax(buy_probabilities[assignment]))
            for assignment in product(range(2), range(3))
        },
    }


def main() -> None:
    """Run the three-method comparison and optionally save the numerical results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[50, 200, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=100000)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    optimum, pyagrum_policy = solve_pyagrum()
    enumerated = exact_optimum()
    assert np.isclose(optimum, enumerated, atol=1e-9, rtol=0.0)
    assert np.isclose(
        exact_policy_value(pyagrum_policy), enumerated, atol=1e-9, rtol=0.0
    )
    print(f"Exact optimum: {optimum:.6f} million (pyAgrum and exhaustive enumeration)")
    diagram = build_pylimid()
    backward = solve(
        diagram,
        method="backward_induction",
        num_samples=20000,
        rng_key=jax.random.PRNGKey(0),
    )
    rows = []

    def record(method, steps, seed, reported, policy):
        value = exact_policy_value(policy)
        row = {
            "method": method,
            "steps": steps,
            "seed": seed,
            "reported_utility": reported,
            "exact_policy_utility": value,
            "exact_regret": optimum - value,
            "argmax_policy_utility": exact_policy_value(deterministic_policy(policy)),
        }
        rows.append(row)
        print(
            f"{method:19s} {steps:>5d} {seed:>4d} {reported:>12.3f} "
            f"{value:>12.3f} {optimum - value:>10.3f} "
            f"{row['argmax_policy_utility']:>12.3f}"
        )

    print(
        "\nMethod              Steps Seed     Reported     Exact EU     Regret"
        "    Argmax EU"
    )
    record("pyAgrum", 0, 0, optimum, pyagrum_policy)
    record("Backward induction", 0, 0, backward.expected_utility, backward.policy)
    trainable = build_pylimid(trainable=True)
    fitted_policies = {}
    for steps in args.steps:
        for seed in args.seeds:
            fitted = solve(
                trainable,
                method="policy_gradient",
                num_steps=steps,
                num_samples=args.batch_size,
                rng_key=jax.random.PRNGKey(seed),
            )
            record(
                "Policy gradient", steps, seed, fitted.expected_utility, fitted.policy
            )
            fitted_policies[steps, seed] = fitted

    selected = fitted_policies[max(args.steps), args.seeds[0]]
    print("\nPolicy probabilities for the first seed at the largest budget:")
    print(selected.render(trainable))
    test_probabilities, buy_probabilities = _probability_tables(selected.policy)
    print(f"P(test=do): {test_probabilities[0]:.8f}")
    print(f"P(buy | not_do, no_result): {buy_probabilities[1, 2, 0]:.8f}")
    values = np.asarray(
        infer(
            trainable,
            ["utility"],
            policy=selected.policy,
            num_samples=args.evaluation_samples,
            rng_key=jax.random.PRNGKey(1_000_000),
        )["utility"].values,
        dtype=np.float64,
    )
    mean = float(values.mean())
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    print(
        f"\nIndependent evaluation ({len(values)} draws): {mean:.3f}, "
        f"approximate 95% interval [{mean - margin:.3f}, {mean + margin:.3f}]"
    )
    print(
        f"The same policy's exact value is {exact_policy_value(selected.policy):.3f}; "
        "this interval measures sampling uncertainty, not optimality."
    )
    print("Impossible rows have zero trajectory probability and do not affect regret.")
    if args.csv is not None:
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
