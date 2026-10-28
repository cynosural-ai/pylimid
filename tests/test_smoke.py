"""Smoke tests verifying the scaffold is importable and wired correctly."""


def test_version_is_set() -> None:
    """The package exposes the version recorded in its distribution metadata."""
    from importlib.metadata import version

    import pylimid

    assert pylimid.__version__ == version("pylimid")


def test_top_level_exports_the_public_api() -> None:
    """The main names import from pylimid directly."""
    import pylimid
    from pylimid import graph, inference, policy, solver
    from pylimid.errors import InferenceError, SolveError
    from pylimid.policy import Policy

    assert pylimid.InfluenceDiagram is graph.InfluenceDiagram
    assert pylimid.ChanceNode is graph.ChanceNode
    assert pylimid.DecisionNode is graph.DecisionNode
    assert pylimid.UtilityNode is graph.UtilityNode
    assert pylimid.infer is inference.infer
    assert pylimid.solve is solver.solve
    assert pylimid.Solution is solver.Solution
    assert pylimid.Posterior is inference.Posterior
    assert pylimid.InferenceResult is inference.InferenceResult
    assert pylimid.InferenceError is InferenceError
    assert pylimid.SolveError is SolveError
    assert pylimid.Policy is Policy
    assert pylimid.ActionTable is policy.ActionTable
    assert pylimid.DecisionRule is policy.DecisionRule
    assert pylimid.FittedPolicy is policy.FittedPolicy
    assert pylimid.DecisionInformation is policy.DecisionInformation
    assert pylimid.PolicyParameters is policy.PolicyParameters
    assert pylimid.PolicyFamily is policy.PolicyFamily
    assert pylimid.LinearSoftmax is policy.LinearSoftmax
    assert pylimid.TabularSoftmax is policy.TabularSoftmax
    assert pylimid.CategoricalPolicy is policy.CategoricalPolicy
    assert pylimid.ContinuousPolicy is policy.ContinuousPolicy
    assert pylimid.DistributionPolicy is policy.DistributionPolicy
    assert pylimid.ParameterModel is policy.ParameterModel
    assert pylimid.Linear is policy.Linear
