"""Sphinx configuration for the pylimid documentation site."""

import os
import sys
from pathlib import Path

from sphinx.util.nodes import make_refnode

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

# ``default_role = "py:obj"`` turns a bare ``Name`` in a docstring into a
# cross-reference, but Sphinx resolves a bare name only in the current
# module or class. Most of our references cross modules (``Snapshot``,
# ``Posterior``, ``InfluenceDiagram.validate``), so a missing-reference
# hook links a name when exactly one documented object matches it.


def resolve_missing_reference(app, env, node, contnode):
    """Link an unqualified Python reference to a unique documented object."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget")
    if not target:
        return None
    matches = env.domains["py"].find_obj(env, "", "", target, None, searchmode=1)
    # A re-exported object is registered under both its public path and its
    # canonical module path; both entries share one target, so collapse them.
    unique = {(entry.docname, entry.node_id): entry for _, entry in matches}
    if len(unique) != 1:
        return None
    entry = next(iter(unique.values()))
    return make_refnode(
        app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, target
    )


def setup(app):
    """Register the missing-reference hook."""
    app.connect("missing-reference", resolve_missing_reference)


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = "pylimid"

# The bundled autosummary class template omits ``:members:``, so a class
# page lists its methods in a summary table without documenting them; the
# table entries then have nothing to link to. The local override adds
# ``:members:``.
templates_path = ["_templates"]

# ``docs/README.md`` documents the build for contributors and is not
# part of the published site.
exclude_patterns = ["_build", "README.md"]
