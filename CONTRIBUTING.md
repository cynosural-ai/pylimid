# Contributing to pylimid

Thanks for your interest in improving pylimid. This guide covers the development setup, the checks a change has to pass, and the conventions the project follows.

## Development setup

pylimid requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cynosural-ai/pylimid
cd pylimid
uv sync --all-groups
pre-commit install
```

`uv sync --all-groups` installs the dev, test, and docs dependency groups. `pre-commit install` wires the lint, type-check, and example-sync hooks.

## Checks

Run these before opening a pull request:

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run ty check               # type check
uv run pytest tests/          # tests
```

Ruff and ty also run on every commit through pre-commit; the Jupytext sync runs on push because it imports JAX. To run the example sync on demand:

```bash
uv run pre-commit run jupytext-examples --hook-stage pre-push --all-files
```

## Repository layout

| Path | Contents |
| --- | --- |
| `pylimid/graph/` | Node types, the mutable `InfluenceDiagram`, validation, and rendering |
| `pylimid/inference/` | The NumPyro engine, solvers, and result types |
| `tests/` | The pytest suite; fully discrete models are cross-checked against pyAgrum |
| `docs/` | The published Sphinx site (see `docs/README.md` for build details) |
| `examples/` | Runnable notebooks, each paired with a Jupytext `.py` twin |
| `design/` | Contributor-facing decision records and design notes; not published |
| `AGENTS.md` | Detailed code style and design conventions |

## Conventions

The full conventions live in [AGENTS.md](AGENTS.md); the highlights:

- Google-style docstrings for every public function, method, and class.
- Public API names in single backticks (`` `Snapshot` ``), literals in double backticks (`` ``None`` ``).
- No comments that reference design notes, ADRs, or ticket numbers; docstrings describe what the code does, not the history of the decision.
- Prefer a loud failure over a silent fallback, and do not defensively re-clean data an earlier step already guarantees.
- The production API comes first: do not add parameters, hooks, or injection points just so tests can override them.

## Documentation

Build the site with:

```bash
uv run sphinx-build -W docs docs/_build/html
```

The `-W` flag turns warnings into errors. Prose pages are Markdown; pages with executable snippets carry `file_format: mystnb` frontmatter and use `{code-cell}` fences. Add a new page to the appropriate `{toctree}` in `docs/index.md`. Development notes for contributors belong in `design/`, not `docs/`.

Examples are paired `.ipynb` / `.py` files. The `.py` side is the source of truth; edit it and let the pre-push hook sync and execute the pair, or sync a single example with:

```bash
uv run jupytext --sync examples/03_influence_diagrams/medical_treatment.py
```

## Pull requests

- Keep each pull request focused on one change.
- Fill in the repository's pull request template: summary, what changed, why, and how it was tested.
- Add or update tests for behavior changes. For discrete models, pyAgrum provides the exact reference.
- Update the docs when the public API or behavior changes.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
