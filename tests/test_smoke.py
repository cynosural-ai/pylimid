"""Smoke tests verifying the scaffold is importable and wired correctly."""


def test_version_is_set() -> None:
    """The package exposes the version recorded in its distribution metadata."""
    from importlib.metadata import version

    import pylimid

    assert pylimid.__version__ == version("pylimid")


def test_top_level_exports_the_public_api() -> None:
    """The main names import from pylimid directly."""
    import pylimid
    from pylimid import graph, inference

    assert pylimid.InfluenceDiagram is graph.InfluenceDiagram
    assert pylimid.ChanceNode is graph.ChanceNode
    assert pylimid.DecisionNode is graph.DecisionNode
    assert pylimid.UtilityNode is graph.UtilityNode
    assert pylimid.infer is inference.infer
    assert pylimid.solve is inference.solve
    assert pylimid.Solution is inference.Solution
    assert pylimid.Posterior is inference.Posterior
