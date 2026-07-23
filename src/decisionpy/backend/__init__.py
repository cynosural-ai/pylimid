"""
Backend layer: translates the graph representation into a runnable model.

Backend-agnostic by construction — nothing in this package is imported by
:mod:`decisionpy.graph`, and this ``__init__`` deliberately imports nothing.
Import a concrete backend module (e.g. :mod:`decisionpy.backend.numpyro`)
explicitly; that is where the backend's heavy dependencies are pulled in.
"""
