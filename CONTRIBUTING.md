# Contributing to pylimid

Thanks for your interest in improving pylimid. This guide covers the development setup, the checks a change has to pass, and the conventions the project follows.

## Development setup

pylimid requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cynosural-ai/pylimid
cd pylimid
uv sync --all-groups
pre-commit install
```

`uv sync --all-groups` installs the dev, test, and docs dependency groups. `pre-commit install` wires the lint and type-check hooks.

## Checks

Run these before opening a pull request:

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run ty check               # type check
uv run pytest tests/          # tests
```

Ruff and ty also run on every commit through pre-commit.

## Repository layout

| Path | Contents |
| --- | --- |
| `pylimid/graph/` | Node types, the mutable `InfluenceDiagram`, validation, and rendering |
| `pylimid/inference/` | Posterior queries and posterior result types |
| `pylimid/solver/` | Joint policy-gradient optimisation and fitted policy results |
| `pylimid/numpyro/` | NumPyro model translation and sampling |
| `pylimid/policy.py`, `pylimid/errors.py` | Shared policy representation and errors |
| `tests/` | Graph, inference, and optimizer tests with analytic utility and gradient references |
| `docs/` | The published Sphinx site (see `docs/README.md` for build details) |
| `design/` | Contributor-facing decision records and design notes; not published |
| `AGENTS.md` | Detailed code style and design conventions |

## Conventions

The full conventions live in [AGENTS.md](AGENTS.md); the highlights:

- Google-style docstrings for every public function, method, and class.
- Public API names in single backticks (`` `DiagramSnapshot` ``), literals in double backticks (`` ``None`` ``).
- No comments that reference design notes, ADRs, or ticket numbers; docstrings describe what the code does, not the history of the decision.
- Prefer a loud failure over a silent fallback, and do not defensively re-clean data an earlier step already guarantees.
- The production API comes first: do not add parameters, hooks, or injection points just so tests can override them.

## Documentation

Build the site with:

```bash
uv run sphinx-build -W docs docs/_build/html
```

The `-W` flag turns warnings into errors. Prose pages are Markdown; pages with executable snippets carry `file_format: mystnb` frontmatter and use `{code-cell}` fences. Add a new page to the appropriate `{toctree}` in `docs/index.md`. Development notes for contributors belong in `design/`, not `docs/`.

## Pull requests

- Keep each pull request focused on one change.
- Fill in the repository's pull request template: summary, what changed, why, and how it was tested.
- Add or update tests for behavior changes. Use analytic expectations or exact sums on small models to verify numerical behavior.
- Update the docs when the public API or behavior changes.

## Releases

Releases are published by the [Release workflow](.github/workflows/release.yml) when a tag starting with `v` is pushed. The package version comes from Git tags through `hatch-vcs`; there is no version number to update manually in `pyproject.toml` or `pylimid/_version.py`.

1. Merge the intended changes into `main`, then update the local checkout:

   ```bash
   git switch main
   git pull --ff-only
   git status --short
   ```

   Confirm the working tree is clean and the [checks](#checks) pass on the release commit.

2. Review the README and documentation, and rebuild the site with every example re-executed:

   ```bash
   uv run sphinx-build -W -E -a -D nb_execution_mode=force docs docs/_build/html
   ```

   See [docs/README.md](docs/README.md) for build requirements and local preview instructions.

3. Create and push an annotated tag. For version 0.2.0:

   ```bash
   git tag -a v0.2.0 -m "Release 0.2.0"
   git push origin v0.2.0
   ```

   Use a new tag for each release. The `v` prefix is required to trigger publishing.

4. Follow the Release workflow in GitHub Actions. It builds the source distribution and wheel, publishes them to PyPI, then creates a GitHub release with the distribution files and automatically generated release notes. Check that all jobs succeed and review the published release notes.

To test publishing first, manually run the Release workflow from GitHub Actions for the intended branch or tag. A manual run publishes to TestPyPI; production PyPI publishing and GitHub release creation run only on tag pushes.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
