# Installation

## Requirements

pylimid requires Python 3.12 (3.13 is not supported yet). NumPyro and JAX are installed as dependencies; the default JAX wheel runs on CPU.

## Install

Install the latest release from PyPI:

```bash
pip install pylimid
```

Or the development version from the repository:

```bash
pip install "pylimid @ git+https://github.com/cynosural-ai/pylimid.git"
```

## CPU and GPU

pylimid runs on whichever device JAX selects; there is no device setting to change. The default install ships the CPU build of JAX.

For an NVIDIA GPU, install the `cuda12` extra (or `cuda13` for newer drivers), which pulls the matching CUDA-enabled JAX plugin:

```bash
pip install "pylimid[cuda12]"
```

You can also add the plugin to an existing install directly with `pip install --upgrade "jax[cuda12]"`. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the full CUDA-version table. Without a CUDA build, JAX prints a one-line notice and falls back to CPU.

## Development setup

For working on pylimid itself, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cynosural-ai/pylimid
cd pylimid
uv sync --all-groups
uv run pytest
```

Build the documentation locally with Sphinx:

```bash
uv run sphinx-build -W docs docs/_build/html
```

The `-W` flag turns warnings into errors, so the build fails on broken references and malformed pages.
