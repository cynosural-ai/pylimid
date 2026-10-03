"""Sphinx configuration for the pylimid documentation site."""

import os
import sys
from pathlib import Path

# Run the executed snippets on CPU during the docs build: the rendered
# outputs are then reproducible, and JAX does not print its "GPU present but
# no CUDA jaxlib" warning on machines that happen to have an NVIDIA card.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "pylimid"
author = "pylimid developers"
copyright = "2026, pylimid developers"

extensions = [
    # myst-nb pulls in myst-parser for Markdown prose.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
]

# Single backticks in docstrings and prose resolve to Python objects. Names
# used across modules need a short missing-reference hook that resolves a
# bare name when it is unique; that hook lands with the API reference.
default_role = "py:obj"

autosummary_generate = True

# Docstrings follow the Google convention (see AGENTS.md).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# MyST extensions used by the prose pages.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# Execute the {code-cell} snippets in the guide and cache their outputs
# between builds.
nb_execution_mode = "cache"
nb_execution_timeout = 600

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = "pylimid"

exclude_patterns = ["_build"]
