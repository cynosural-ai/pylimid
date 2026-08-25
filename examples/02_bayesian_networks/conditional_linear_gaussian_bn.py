# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %%
"""
Conditional linear-Gaussian Bayesian networks — PLACEHOLDER, not implemented yet.

A conditional linear-Gaussian BN is a mixed network following the CLG
convention:

- discrete variables may have only discrete parents;
- continuous variables may have both discrete and continuous parents, and
  their distribution is Gaussian with a mean linear in the continuous parents
  and a per-discrete-assignment offset/scale,

    y | (g, x) ~ Normal(loc = a[g] + b * x, scale = sigma[g])

with g discrete and x continuous. Example shape: a discrete root g feeding a
continuous child y (one mean and scale per g state), which in turn feeds a
continuous child z linear in y.

Current status:
- The models themselves can be built with the existing graph layer and run
  through the NumPyro engine (approximate Monte-Carlo draws; every query
  variable comes back as a Posterior — raw draws plus states, with
  `.marginal()` for the discrete nodes' probability vectors).
- Exact CLG inference (Lauritzen-style propagation: a Gaussian posterior per
  discrete assignment, a Gaussian mixture once the discrete states are
  marginalized) is not implemented yet. The discrete-parents-only constraint
  is a modeling convention today, not something the library enforces. This
  example notebook will land with that engine.
"""
