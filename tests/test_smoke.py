"""Smoke tests verifying the scaffold is importable and wired correctly."""


def test_version_is_set() -> None:
    """The package exposes a version string."""
    import decisionpy

    assert decisionpy.__version__ == "0.1.0"


def test_top_level_exports_the_public_api() -> None:
    """The main names import from decisionpy directly."""
    import decisionpy
    from decisionpy import graph, inference

    assert decisionpy.InfluenceDiagram is graph.InfluenceDiagram
    assert decisionpy.ChanceNode is graph.ChanceNode
    assert decisionpy.DecisionNode is graph.DecisionNode
    assert decisionpy.UtilityNode is graph.UtilityNode
    assert decisionpy.infer is inference.infer
    assert decisionpy.solve is inference.solve
    assert decisionpy.Solution is inference.Solution
    assert decisionpy.Posterior is inference.Posterior
