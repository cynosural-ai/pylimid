# Installation

## Requirements

pylimid requires Python 3.12 (3.13 is not supported yet). NumPyro and JAX are installed as dependencies; the default JAX wheel runs on CPU.

## Install from source

pylimid is not on PyPI yet. Install the latest development version with pip:

```bash
pip install "pylimid @ git+https://github.com/cynosural-ai/pylimid.git"
```

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

## CPU and GPU

pylimid runs on whatever device JAX selects; there is no device setting to change. The default `jax` dependency installs the CPU build, and JAX uses a GPU automatically when a CUDA-enabled JAX is installed.

For GPU support, install the JAX build that matches your CUDA version:

```bash
pip install --upgrade "jax[cuda12]"
```

Use `cuda13` if that is the version your driver supports; the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) has the full table. Without a CUDA build, JAX prints a one-line notice and falls back to CPU.
