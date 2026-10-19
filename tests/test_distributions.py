"""Chance-distribution contracts across inference and solver execution paths."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pylimid import (
    ChanceNode,
    DecisionNode,
    InferenceError,
    InfluenceDiagram,
    UtilityNode,
    infer,
    solve,
)
from pylimid.solver.backward_induction import solve as backward_induction_solve
from pylimid.solver.scan.batched import solve as batched_solve
from pylimid.solver.scan.intervention import solve as intervention_solve


@pytest.fixture(
    params=[
        pytest.param(
            lambda p: dist.Categorical(probs=jnp.stack((1 - p, p), axis=-1)),
            id="categorical-probs",
        ),
        pytest.param(
            lambda p: dist.Categorical(logits=jnp.log(jnp.stack((1 - p, p), axis=-1))),
            id="categorical-logits",
        ),
        pytest.param(lambda p: dist.Bernoulli(probs=p), id="bernoulli-probs"),
        pytest.param(
            lambda p: dist.Bernoulli(logits=jnp.log(p) - jnp.log1p(-p)),
            id="bernoulli-logits",
        ),
    ]
)
def binary_distribution(request):
    """Construct either supported binary distribution in both parameterizations."""
    return request.param


def _decision_diagram(distribution):
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(
            name="weather", states=("dry", "wet"), dist=lambda: distribution(0.5)
        )
    )
    diagram.add_node(
        DecisionNode(name="action", parents=("weather",), states=("no", "yes"))
    )
    diagram.add_node(
        ChanceNode(
            name="outcome",
            parents=("weather", "action"),
            states=("bad", "good"),
            dist=lambda weather, action: distribution(
                0.1 + 0.7 * weather + 0.1 * action
            ),
        )
    )
    diagram.add_node(
        UtilityNode(
            name="payoff",
            parents=("weather", "action", "outcome"),
            values=lambda weather, action, outcome: (
                10.0 * (weather == action) + outcome
            ),
        )
    )
    return diagram


@pytest.fixture(params=["inference", "backward_induction", "batched", "intervention"])
def execution(request):
    """Run a diagram through inference or one of the three simulation walks."""
    return request.param


def _execute(diagram, execution):
    if execution == "inference":
        return infer(
            diagram,
            ["weather", "outcome", "payoff"],
            policy={"action": {(0,): 0, (1,): 1}},
            num_samples=256,
        )
    solver = {
        "backward_induction": backward_induction_solve,
        "batched": batched_solve,
        "intervention": intervention_solve,
    }[execution]
    return solver(diagram.snapshot(), num_samples=256)


def test_supported_distributions_with_chance_and_decision_parents(
    binary_distribution, execution
):
    """Parent-dependent distributions work inside each execution's JAX transforms."""
    result = _execute(_decision_diagram(binary_distribution), execution)
    if execution == "inference":
        assert result["weather"].marginal() == pytest.approx([0.5, 0.5], abs=0.1)
        assert jnp.array_equal(result["payoff"].values, 10 + result["outcome"].values)
    else:
        assert result.policy == {"action": {(0,): 0, (1,): 1}}
        assert result.expected_utility == pytest.approx(10.5, abs=0.2)


def test_supported_distributions_during_posterior_enumeration(binary_distribution):
    """Evidence updates weather through a chance- and decision-dependent likelihood."""
    result = infer(
        _decision_diagram(binary_distribution),
        ["weather", "outcome", "payoff"],
        policy={"action": {(0,): 0, (1,): 1}},
        observed={"outcome": 1},
        num_samples=1000,
    )
    assert result["weather"].marginal() == pytest.approx([0.1, 0.9], abs=0.04)
    assert jnp.all(result["outcome"].values == 1)
    assert jnp.all(result["payoff"].values == 11)


@pytest.mark.parametrize(
    "states,distribution,message",
    [
        (("dry",), lambda: dist.Categorical(probs=jnp.array([0.5, 0.5])), "2 outcomes"),
        (
            ("dry", "wet", "snow"),
            lambda: dist.Categorical(logits=jnp.array([0.0, 0.0])),
            "2 outcomes",
        ),
        (
            tuple(str(i) for i in range(6)),
            lambda: dist.Poisson(3.0),
            "Poisson.*require Categorical or Bernoulli",
        ),
    ],
    ids=["too-few-labels", "too-many-labels", "poisson-finite-states"],
)
def test_invalid_policy_table_parent_fails_before_sampling(
    execution, states, distribution, message
):
    """Invalid category codes must never reach a decision's policy table."""
    diagram = _decision_diagram(lambda p: dist.Bernoulli(probs=p))
    diagram["weather"].states = states
    diagram["weather"].dist = distribution
    # Structural validation does not inspect distribution factories.
    assert diagram.validate() == []
    if execution == "inference":
        policy = {"action": {(i,): 0 for i in range(len(states))}}
        with pytest.raises(ValueError, match=f"weather.*{message}"):
            infer(diagram, ["payoff"], policy=policy, num_samples=10)
    else:
        with pytest.raises(ValueError, match=f"weather.*{message}"):
            _execute(diagram, execution)


@pytest.mark.parametrize("states", [("only",), ("a", "b", "c")])
def test_binary_distribution_requires_exactly_two_labels(binary_distribution, states):
    """Cardinality checks cover both distribution families and parameterizations."""
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="x", states=states, dist=lambda: binary_distribution(0.4))
    )
    with pytest.raises(ValueError, match="x.*2 outcomes.*exactly 2 labels"):
        infer(diagram, ["x"], num_samples=10)


@pytest.mark.parametrize("logits", [False, True], ids=["probs", "logits"])
@pytest.mark.parametrize("label_count", [2, 3])
def test_zero_probability_categories_still_require_labels(logits, label_count):
    """A zero-mass category remains part of the declared category domain."""
    diagram = InfluenceDiagram()

    def distribution():
        if logits:
            return dist.Categorical(logits=jnp.array([0.0, 0.0, -jnp.inf]))
        return dist.Categorical(probs=jnp.array([0.5, 0.5, 0.0]))

    diagram.add_node(
        ChanceNode(
            name="x",
            states=tuple(str(i) for i in range(label_count)),
            dist=distribution,
        )
    )
    if label_count == 2:
        with pytest.raises(ValueError, match="x.*3 outcomes.*exactly 3 labels"):
            infer(diagram, ["x"], num_samples=10)
    else:
        posterior = infer(diagram, ["x"], num_samples=100)["x"]
        assert posterior.marginal()[2] == 0


@pytest.mark.parametrize("with_evidence", [False, True], ids=["forward", "posterior"])
def test_supported_discrete_distributions_require_labels(
    binary_distribution, with_evidence
):
    """Reject missing labels on both forward and continuous-posterior paths."""
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="x", dist=lambda: binary_distribution(0.4)))
    diagram.add_node(
        ChanceNode(name="report", parents=("x",), dist=lambda x: dist.Normal(x, 1.0))
    )
    with pytest.raises(ValueError, match="x.*states=None.*discrete.*matching states"):
        infer(
            diagram,
            ["x"],
            observed={"report": 0.0} if with_evidence else None,
            num_samples=10,
        )


@pytest.mark.parametrize(
    "distribution",
    [lambda: dist.Poisson(3.0), lambda: dist.Binomial(total_count=1, probs=0.5)],
    ids=["poisson", "binomial"],
)
def test_count_distributions_cannot_declare_categories(distribution):
    """State labels cannot turn numerical counts into categorical outcomes."""
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="count", states=("low", "high"), dist=distribution)
    )
    with pytest.raises(ValueError, match="count.*require Categorical or Bernoulli"):
        infer(diagram, ["count"], num_samples=10)


def test_continuous_distribution_cannot_have_state_labels():
    diagram = InfluenceDiagram()
    diagram.add_node(
        ChanceNode(name="x", states=("low", "high"), dist=lambda: dist.Normal(0.0, 1.0))
    )
    with pytest.raises(
        ValueError, match="x.*Normal.*Numerical distributions require states=None"
    ):
        infer(diagram, ["x"], num_samples=10)


@pytest.mark.parametrize("observed_node", ["weather", "outcome"])
@pytest.mark.parametrize("states", [None, ("only",)], ids=["unlabeled", "mismatched"])
def test_observed_distributions_are_validated(observed_node, states):
    diagram = _decision_diagram(lambda p: dist.Bernoulli(probs=p))
    diagram[observed_node].states = states
    message = "2 outcomes" if states else "states=None.*discrete"
    with pytest.raises(ValueError, match=f"{observed_node}.*{message}"):
        infer(diagram, ["outcome"], observed={observed_node: 0}, policy={"action": 0})


@pytest.mark.parametrize(
    "states", [None, ("bad", "good")], ids=["unlabeled", "mismatched"]
)
def test_parent_dependent_mismatch_is_rejected(execution, states):
    diagram = _decision_diagram(lambda p: dist.Bernoulli(probs=p))
    diagram["outcome"].states = states
    diagram["outcome"].dist = lambda weather, action: dist.Categorical(
        probs=jnp.stack(
            (0.1 + 0.1 * weather, 0.1 + 0.1 * action, 0.8 - 0.1 * (weather + action)),
            axis=-1,
        )
    )
    message = "3 outcomes.*exactly 3 labels" if states else "states=None.*discrete"
    with pytest.raises(ValueError, match=f"outcome.*{message}"):
        _execute(diagram, execution)


@pytest.mark.parametrize("method", ["backward_induction", "scan"])
def test_public_solve_uses_existing_inference_error_convention(method):
    diagram = _decision_diagram(lambda p: dist.Bernoulli(probs=p))
    diagram["weather"].states = ("only",)
    with pytest.raises(InferenceError, match="weather.*2 outcomes"):
        solve(diagram, method=method, num_samples=10)


def test_invalid_unrelated_distribution_remains_pruned():
    diagram = InfluenceDiagram()
    diagram.add_node(ChanceNode(name="x", dist=lambda: dist.Normal(0.0, 1.0)))
    diagram.add_node(
        ChanceNode(
            name="unused", dist=lambda: dist.Categorical(probs=jnp.array([0.5, 0.5]))
        )
    )
    assert infer(diagram, ["x"], num_samples=10)["x"].values.shape == (10,)
    with pytest.raises(ValueError, match="unused.*Categorical"):
        infer(diagram, ["unused"], num_samples=10)
