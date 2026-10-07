# Installation

PyLIMID uses Python 3.12. Installing it also pulls in NumPyro and JAX, so there is nothing else to set up.

## Install

To get the latest release from PyPI:

```bash
pip install pylimid
```


## Running on a GPU

PyLIMID runs on whichever device JAX picks. By default, it is installed in its CPU-only version. So, if you have an NVIDIA GPU, install the extra that matches your CUDA version, which pulls in the CUDA build of JAX:

```bash
pip install "pylimid[cuda12]"  # CUDA 12
pip install "pylimid[cuda13]"  # CUDA 13, for newer drivers
```

If PyLIMID is already installed, you can also add GPU support by upgrading JAX directly:

```bash
pip install --upgrade "jax[cuda12]"
```

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) explains which CUDA version matches your drivers. If JAX finds a GPU but no CUDA build is installed, it prints a short warning and keeps running on CPU.