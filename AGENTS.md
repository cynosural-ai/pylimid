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
- Cross-references in docstrings are plain text: no Sphinx roles (`:class:`, `:meth:`, `:func:`, `:attr:`, `:mod:`). Write the name bare (`Snapshot`, `validate()`) without backticks or roles.

Example:
```python
def add_edge(self, parent: str, child: str) -> None:
    """
    Wire an edge parent -> child.

    Both endpoints must already be in the diagram.

    Args:
        parent: Name of the parent node.
        child: Name of the child node.

    Raises:
        KeyError: If either endpoint is not in the diagram.
        ValueError: If the edge would create a cycle or the child is a utility node.
    """
```

## Comments and docs in code
- Do not reference intermediate design notes, ADRs-in-progress, option letters (A/B/C), ticket numbers, or `docs/*.md` files from code comments or docstrings. Those files are temporary and will confuse readers later.
- Docstrings describe what the code does and its inputs/outputs — not the history of the decision.

## Production API over testability
- Do not add parameters, hooks, or injection points "so tests can override." Production callers define the API; tests adapt (fixtures, monkeypatch) if needed. This does not forbid dependency injection that production itself needs.
- Do not keep unused helpers or constants only for tests.

## Defensive code
Don't add defensive code for cases that can't happen. If data is already cleaned upstream (by a SQL `WHERE`, a model output contract, a config enum, or an earlier step in this codebase), don't re-clean it in Python — let bad data fail loudly instead of silently coercing it. Silent fallbacks (`errors="coerce"`, `fillna`, `np.where(..., default)`, `dict.get(key, default)`) hide real problems; a loud `ValueError` (or `NotImplementedError` while scaffolding) surfaces them.

Do not wrap obvious failures in friendlier exceptions "just in case" (e.g. catching `KeyError` to re-raise `ValueError`). Prefer the natural failure.

Example — remove this:
```python
# `marginal()` already summed out every variable not in `target_order`,
# so the lookup below can never miss.
old_lookup = {v: assignment[target_order.index(v)] for v in target_order}
old_tuple = tuple(old_lookup.get(v, 0) for v in self.variables)
```

Just index the dict — the invariant makes the default unreachable, and if it ever breaks you want a loud `KeyError`, not silently wrong probabilities:
```python
old_tuple = tuple(old_lookup[v] for v in self.variables)
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

