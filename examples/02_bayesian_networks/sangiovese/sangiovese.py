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

# %% [markdown]
# # SANGIOVESE — a realistic conditional linear Gaussian network
#
# A mixed network fitted on real agronomic data: whether a specific
# agronomic *treatment* was applied (a 16-level discrete root) shapes a
# cascade of continuous vine-and-grape measurements, ending in the
# grape-quality variables (GrapeW, Brix, pH, Anthoc, Polyph). Nodes with
# Treatment among their parents are conditional linear Gaussian — a Normal
# with a per-treatment mean, and for some, a per-treatment regression
# coefficient on their Gaussian parents.
#
# Source: A. Magrini, S. Di Blasi and F. M. Stefanini (2017), *A
# conditional linear Gaussian network to assess the impact of several
# agronomic settings on the quality of Tuscan Sangiovese grapes*,
# Biometrical Letters 54(1):25-42. Model downloaded from the
# [bnlearn Bayesian Network Repository](https://www.bnlearn.com/bnrepository/)
# (CC BY-SA 3.0): SANGIOVESE, 15 nodes, 55 arcs, 259 parameters. The
# parameters live in `_sangiovese.py`, extracted from the repository's
# `sangiovese.rda`.
#
# The engine runs the NumPyro mixed path: NUTS over the continuous nodes
# with the discrete Treatment enumerated out — every result is a
# Monte-Carlo estimate (no exact reference; pyAgrum's CLG module is
# pure-Gaussian, so there is nothing exact to compare against for a mixed
# model like this one).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples" / "02_bayesian_networks" / "sangiovese"))

from _sangiovese import TREATMENT_LEVELS, TREATMENT_PRIOR, pylimid_diagram

from pylimid.inference import infer

# %% [markdown]
# ## The model
#
# `Treatment` is the only discrete node — the 16 agronomic treatments of
# the trial. Four nodes depend on it directly (SproutN, BunchN, SPAD06,
# Brix); the rest are plain linear Gaussian, so every variable downstream
# of Treatment is a mixture of Gaussians driven by the treatment prior.

# %%
diag = pylimid_diagram()
print(diag.validate())
snapshot = diag.snapshot()
print("nodes:", len(snapshot.nodes), "| treatments:", len(TREATMENT_LEVELS))
print("treatment prior (first five):", [round(p, 3) for p in TREATMENT_PRIOR[:5]])

# %% [markdown]
# ## The network
#
# The structure, rendered with Mermaid (native in JupyterLab ≥ 4.1 and
# Notebook ≥ 7): Treatment feeds SproutN, BunchN, SPAD06 and Brix
# directly; the rest is a dense web of vine measurements converging on
# the quality leaves (GrapeW, Brix, pH, Anthoc, Polyph).

# %%
from IPython.display import Markdown

Markdown(f"```mermaid\n{diag.to_mermaid()}\n```")

# %% [markdown]
# ## The prior
#
# The quality variables are standardized (mean ≈ 0, sd ≤ 1). The prior is
# a mixture over the 16 treatments, so its spread reflects both within-
# and between-treatment variability.

# %%
quality = ["GrapeW", "Brix", "Anthoc", "Polyph", "Acid", "pH"]
prior = infer(diag, quality)
for name in quality:
    p = prior[name]
    print(f"{name:8s} mean={p.mean():+.3f}  sd={p.std():.3f}  hdi=({p.hdi()[0]:+.3f}, {p.hdi()[1]:+.3f})")

# %% [markdown]
# ## Inverting the model: which treatment produced this quality?
#
# The forward model is `treatment -> measurements`. We can also query it
# backwards: given an observed measurement profile, what does the model
# believe about the treatment? The treatment prior is almost uniform
# (each level ≈ 0.06), so any concentration in the posterior is evidence
# from the measurements themselves.
#
# First query: high grape weight on its own. Weak signal — no treatment
# rises much above the prior.

# %%
post_w = infer(diag, ["Treatment"], observed={"GrapeW": 0.6})
top5 = sorted(enumerate(post_w["Treatment"].marginal()), key=lambda x: -x[1])[:5]
print("P(Treatment | GrapeW=0.6):")
print("   ", [(TREATMENT_LEVELS[i], round(p, 3)) for i, p in top5])

# %% [markdown]
# ## Adding the joint pattern concentrates the posterior
#
# High grape weight *and* high Brix together pick out the "b" treatment
# cluster sharply: T3b goes from ≈0.06 prior to ≈0.29 — the model reads
# the joint pattern, not the single measurement.

# %%
post_wb = infer(diag, ["Treatment"], observed={"GrapeW": 0.6, "Brix": 0.15})
top5 = sorted(enumerate(post_wb["Treatment"].marginal()), key=lambda x: -x[1])[:5]
print("P(Treatment | GrapeW=0.6, Brix=0.15):")
print("   ", [(TREATMENT_LEVELS[i], round(p, 3)) for i, p in top5])

# %% [markdown]
# ## Contradictory evidence points elsewhere
#
# Now the same grape weight but *low* Brix — a profile the model finds
# unusual. The posterior moves to a different cluster (the "a" treatments,
# e.g. T7a, T1a, T5a): with the joint structure, the model distinguishes
# which treatments plausibly produce which combinations, not just single
# high values.

# %%
post_contr = infer(diag, ["Treatment"], observed={"GrapeW": 0.6, "Brix": -0.08})
top5 = sorted(enumerate(post_contr["Treatment"].marginal()), key=lambda x: -x[1])[:5]
print("P(Treatment | GrapeW=0.6, Brix=-0.08):")
print("   ", [(TREATMENT_LEVELS[i], round(p, 3)) for i, p in top5])

# %% [markdown]
# ## The quality variables under the favoured treatment
#
# Conditioning on the treatment the joint query favoured (T3b) shifts the
# continuous quality posteriors — the forward direction of the model.
# The shifts are honest about the model's structure: Brix (a direct
# child of Treatment) moves clearly, while GrapeW — whose path to
# Treatment runs through several intermediates — barely changes, which is
# exactly why a single high GrapeW was such a weak treatment signal.

# %%
t3b = TREATMENT_LEVELS.index("T3b")
post_t3b = infer(diag, quality, observed={"Treatment": t3b})
for name in quality:
    p = post_t3b[name]
    pr = prior[name]
    print(f"{name:8s} prior mean={pr.mean():+.3f} -> T3b mean={p.mean():+.3f}  (sd {p.std():.3f})")

# %% [markdown]
# ## What the comparison says
#
# - The treatment posterior concentrates on the measurements' joint
#   signal: single measurements barely move it, combinations move it a
#   lot (T3b: 0.06 -> 0.29).
# - The model is a full joint distribution: contradictory evidence
#   (weight high, Brix low) is explained by a *different* treatment
#   cluster, not by the single best treatment.
# - Every number here is a Monte-Carlo estimate (NUTS, 2000 draws): the
#   treatment probabilities wobble by a couple of points run-to-run, and
#   the means/sds shown for continuous nodes carry the same MC noise.
