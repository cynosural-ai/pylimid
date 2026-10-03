# Sphinx documentation — state and handoff

This note is for whoever picks up the documentation work next (possibly on another machine). It records what exists, the decisions already made and why, and the concrete next steps. It complements `documentation_plan.md`, which is the higher-level plan; this file is the working state.

## Where the work is

- Branch: `docs/getting-started`, created on top of `refactor/roadmap` (which is 12 commits ahead of `main` and not yet merged).
- Everything is **uncommitted**. Before switching machines, commit and push this branch; otherwise the next session starts from an empty `docs/`.
- The remote branch `origin/docs/sphinx-site` is unrelated and stale: it is the old `docs/` living notes, moved to `design/` in commit `f2f6709`. It can be deleted.
- Once `refactor/roadmap` merges to `main`, rebase this branch onto `main`.

## What exists now

```
docs/
  conf.py
  index.md                         # landing page
  getting_started/
    index.md                       # section landing + toctree
    installation.md
    quickstart.md                  # executable MyST text notebook
```

- `pyproject.toml`: new `docs` dependency group, added to `[tool.uv] default-groups`.
- `.gitignore`: `docs/_build/`.

`docs/` holds only published Sphinx sources. Development notes stay in `design/` and are never built into the site.

## Toolchain

Installed via the `docs` group: `sphinx`, `myst-nb`, `myst-parser` (pulled by myst-nb), `sphinx-rtd-theme`, `sphinx-copybutton`, `sphinx-design`, `ipykernel`. Resolved versions at the time of writing: Sphinx 8.2.3, myst-nb 1.4.0, myst-parser 5.1.0, sphinx-rtd-theme 3.1.0, sphinx-design 0.7.0, Python 3.12.

Sphinx is pinned `>=8.2,<9` in `pyproject.toml` with a comment. Reason: myst-parser 5.1.0 crashes at setup on Sphinx 9 because it removes `SphinxUnreferencedFootnotesDetector` from the transform registry, and that name no longer exists (`UnreferencedFootnotesDetector` does). Remove the pin once myst-parser supports Sphinx 9.

## Build and serve

```bash
uv sync                                              # installs dev/test/docs groups
uv run sphinx-build -W --keep-going docs docs/_build/html
uv run python -m http.server -d docs/_build/html 8000   # http://localhost:8000
```

`-W` treats warnings as errors; the build is currently clean from scratch. For live reload, `sphinx-autobuild` can be added to the `docs` group later.

## Configuration decisions

- `extensions` in `conf.py` lists `myst_nb` but **not** `myst_parser`. Listing both registers `sub-ref` and `figure-md` twice and raises warnings that fail `-W`. myst-nb pulls in myst-parser.
- `default_role = "py:obj"`, so single backticks in docstrings and prose resolve to Python objects. Bare names used across modules will not resolve without the missing-reference hook described below; that is deferred until the API reference exists, since there is nothing to resolve against yet.
- MyST extensions enabled: `colon_fence`, `deflist`, `dollarmath`, `fieldlist`, `tasklist`. Heading anchors to depth 3.
- `nb_execution_mode = "cache"` (cache lives in `docs/_build/.jupyter_cache`, gitignored). A clean build executes the snippets; a second build reuses the cache. Outputs in the HTML are executed on the build machine.
- `conf.py` sets `JAX_PLATFORMS=cpu` before anything imports JAX. This keeps the rendered outputs reproducible and avoids the `An NVIDIA GPU may be present ... Falling back to cpu` notice appearing in the pages on machines that have a GPU. RTD has no GPU anyway.
- Theme: `sphinx_rtd_theme`, `html_title = "pylimid"`.

## How pages are meant to be written

- **Prose/concept pages**: plain MyST Markdown (`.md`). Examples shown as executable snippets via `{code-cell}` directives where real output matters, otherwise static fenced code.
- **Executable `.md` pages** need YAML top-matter, or myst-nb emits `Found an unexpected code-cell ... [mystnb.nbcell]`:
  ```markdown
  ---
  file_format: mystnb
  kernelspec:
    name: python3
  ---
  ```
  `docs/getting_started/quickstart.md` is the reference example.
- **Full examples** later become `.ipynb` rendered as pages; concept pages keep short snippets and link to them.
- Cross-references: single backticks for public names (resolved via `py:obj` + the future hook), double backticks for literal code/values, per `AGENTS.md`. No explicit Sphinx roles.
- Lines flow; do not hard-wrap at 80 characters.

## Site structure agreed so far

```
Home
Getting started
  Installation
  Quickstart
Concepts
  Building a diagram        (workspace, node types, validation/snapshots)
  Chance nodes              (NumPyro callables; discrete, continuous, CLG)
  Decisions and information sets
  Utilities
  Inference                 (infer, Posterior, references)
  Solving                   (solve, solvers, solvability, Monte-Carlo error, references)
  Scope and limitations
Examples                    (later; grouped BN / ID, incl. exact-vs-Monte-Carlo)
API reference
References                  (bibliography)
Project status              (beta, API-stability promise, unsupported, versions)
```

Key decisions behind this:

- Organize around the two questions the library answers — `infer` ("what does the data say?") and `solve` ("what should I do?"). Do **not** create top-level "Bayesian networks" and "Influence diagrams" doc sections; a BN is just a diagram with no decisions, and the BN/ID split belongs in the Examples grouping.
- Do **not** mirror the API reference with a "fundamental components" section: concept pages explain when/why, the reference lists signatures.
- The discrete vs. continuous decision distinction is a dimension *inside* pages, not a top-level partition. Future continuous decisions should slot into "Decisions and information sets" and "Solving" without moving pages.
- No links to the external blog series. Background pages (when written) stand alone and cite papers via the bibliography.
- Keep the toctree shallow and section names renameable while the API is unstable; adding pages later is cheap.

## Next steps, in order

1. **API reference.** Add `docs/api/` with autosummary pages for the public surface: top-level `pylimid`, `pylimid.graph`, `pylimid.inference`, `pylimid.inference.numpyro.solvers`, filtering members to each module's `__all__`. Public top-level names: `ChanceNode`, `DecisionNode`, `UtilityNode`, `InfluenceDiagram`, `infer`, `solve`, `Solution`, `Posterior`, `Policy`, `InferenceResult`, `InferenceError`, `SolveMethod`, `SolverName`. Then add the bare-name missing-reference hook so `InferenceDiagram`-style single-backtick refs resolve. Recommended shape: a small `docs/_ext/` extension (testable, keeps `conf.py` readable) that, on `missing-reference` with `reftype == "obj"` and a dotted-free target, finds documented Python objects whose final component matches and picks the one with the fewest dots (so `pylimid.solve` wins over `pylimid.inference.engine.solve`); return `None` if still ambiguous.
2. **Project status / scope page.** Beta note, no API-stability promise before 0.1, what is not supported yet (continuous decisions, exact inference), supported Python/JAX versions, where to report issues.
3. **Concepts pages.** Start with "Building a diagram" and "Chance nodes"; Markdown with executable snippets, shared toy models imported from a helper to avoid duplication and JAX cost.
4. **Examples wiring.** Decide how notebooks enter the docs tree. A Sphinx toctree cannot reference files outside the docs source directory (still true through Sphinx 9). Options: symlink `docs/examples/*.ipynb` to `../../examples/...` with `nb_execution_mode = "off"` (fast; watch `.py`/`.ipynb` drift, e.g. a `jupytext --sync` check); symlink the `.py` percent twins and execute on build; or move canonical sources under `docs/`. The examples currently still call `pylimid.inference.numpyro.solvers.batched_solve(diag.snapshot())` and must move to `pylimid.solve` when ported.
5. **RTD.** Add `.readthedocs.yaml` (Python 3.12, install `--group docs`), a `docs` CI job that runs `sphinx-build -W`, and publish a pre-release (`0.1.0b1`) so `pip install pylimid` in the docs is real. Note `stable` ignores pre-releases.
6. **References.** Add `sphinxcontrib-bibtex` and `docs/references.bib` when the first citations appear; the candidate list is in `documentation_plan.md`.
7. **Diagrams (later).** Graphviz `to_dot()` / `_repr_svg_` for static model pictures; conceptual figures under `docs/_static/`.

## Known gotchas

- Building `-W` from a clean tree is the gate; keep it warning-free.
- Do not add `myst_parser` back to `extensions` alongside `myst_nb`.
- Executable Markdown needs the `file_format: mystnb` front matter.
- The notebook cache makes a rebuild fast but can mask a stale execution; delete `docs/_build` to force re-execution.
- `docs/conf.py` forces CPU JAX for the build; that is intentional and does not affect library users.
