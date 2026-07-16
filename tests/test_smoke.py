"""Smoke tests verifying the scaffold is importable and wired correctly."""


def test_version_is_set() -> None:
    """The package exposes a version string."""
    import decisionpy

    assert decisionpy.__version__ == "0.1.0"
