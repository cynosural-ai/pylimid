# Documentation

The published site is built from this directory with [Sphinx](https://www.sphinx-doc.org/) and [myst-nb](https://myst-nb.readthedocs.io/). Prose pages are Markdown. Pages with executable snippets carry `file_format: mystnb` frontmatter and use `{code-cell}` fences; myst-nb executes those cells through a Jupyter kernel and injects their outputs into the page.

## Build

The docs dependencies are in the `docs` dependency group, which `uv` installs by default. Build with:

```bash
uv run sphinx-build -W docs docs/_build/html
```

Open `docs/_build/html/index.html`. The `-W` flag turns warnings into errors, so the build fails on broken references, malformed pages, or a code cell that raises.

Rendering the model figures (the examples' `_repr_svg_` cells) needs the Graphviz `dot` binary on `PATH` (`brew install graphviz`, `apt install graphviz`). Without it the build still succeeds, but the diagrams show their plain repr instead of a figure.

## Preview locally

Serve the built site over HTTP:

```bash
uv run python -m http.server -d docs/_build/html 7777
```

Then open <http://localhost:7777>.

## Force a full rebuild

`nb_execution_mode` is `"cache"` (see `conf.py`), so a code cell is executed only when its source changed since the last build; the results live in `docs/_build/.jupyter_cache`. An incremental build is fast but can leave a cell unexecuted when the library changes underneath it.

To ignore the Sphinx environment and the execution cache and re-run every code cell:

```bash
uv run sphinx-build -W -E -a -D nb_execution_mode=force docs docs/_build/html
```

Delete `docs/_build` as well if you want the generated HTML rebuilt entirely from scratch.

## Notes

- List `myst_nb` in `extensions`, not `myst_parser`: myst-nb pulls myst-parser in, and registering both repeats directives and fails `-W`.
- Sphinx is pinned to `<9` in `pyproject.toml` because myst-parser 5.1 crashes on Sphinx 9. Drop the pin once myst-parser supports it.
- Executable Markdown pages need the `file_format: mystnb` and `kernelspec` top matter (see the quickstart) or myst-nb warns about an unexpected code cell.
- `conf.py` sets `JAX_PLATFORMS=cpu` before JAX loads: build outputs stay reproducible and notebook cells do not print the GPU fallback notice.
- `-W` is the gate; a clean build is warning-free.
- The `examples/` notebooks are kept in sync by the pre-push jupytext hook. Run it on demand with `uv run pre-commit run jupytext-examples --hook-stage pre-push --all-files`.

