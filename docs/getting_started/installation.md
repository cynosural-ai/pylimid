# Installation

PyLIMID uses Python 3.12. Installing it also pulls in NumPyro and JAX.

## Install

To get the latest release from PyPI:

```bash
pip install pylimid
```

```{note}
Drawing diagrams requires [Graphviz](https://graphviz.org/download/): install it with `brew install graphviz` on macOS or `sudo apt install graphviz` on Debian/Ubuntu.
```

## Running on a GPU

The default installation uses the CPU. For NVIDIA GPUs on Linux (x86_64 or aarch64), choose a CUDA extra compatible with your GPU and NVIDIA driver:

```bash
pip install "pylimid[cuda12]"  # CUDA 12
pip install "pylimid[cuda13]"  # CUDA 13
```

If PyLIMID is already installed, you can also add GPU support by upgrading JAX directly:

```bash
pip install --upgrade "jax[cuda12]"
```

These extras install the CUDA libraries automatically. They do not support native Windows or macOS. GPU support under Windows WSL2 is experimental. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for GPU and driver requirements.
