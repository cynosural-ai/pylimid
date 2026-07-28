# decisionpy

## Build, lint, test
```bash
uv run ruff check .        # lint
uv run ruff format --check .  # format check
uv run ty check            # type check
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

## Writing Markdown files
Dont break lines at 80 characters, let lines flow and the viewer will adjust 


## Philosophy
Follow the Zen of Python:

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

