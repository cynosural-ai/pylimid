# decisionpy

## Build, lint, test
```bash
uv run ruff check .        # lint
uv run ruff format --check .  # format check
uv run ty .                # type check
uv run pytest tests/       # run tests
```

## Code style
- Google-style docstrings for all public functions, methods, and classes.

Example:
```python
def add_edge(self, parent: str, child: str) -> None:
    """
    Wire an edge ``parent -> child``.

    Both endpoints must already be in the diagram.

    Args:
        parent: Name of the parent node.
        child: Name of the child node.

    Raises:
        KeyError: If either endpoint is not in the diagram.
        ValueError: If the edge would create a cycle or the child is a utility node.
    """
```

