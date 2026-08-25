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
# # Bayesian networks — inference with infer()
#
# A Bayesian network is an influence diagram with only chance nodes. Once the
# diagram is built (see `01_model_creation/workspace_and_validation.ipynb`),
# inference is one call: `infer()` auto-dispatches to the exact engine when it
# can.
#
# This notebook walks the smallest meaningful model: `rain -> wet_grass`.

# %%
import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, InfluenceDiagram
from decisionpy.inference import infer

# %% [markdown]
# ## The model
#
# Two chance nodes, wired with one edge. The `dist` of `wet_grass` names its
# parent `rain` and returns one row of probabilities per rain state:
# `P(wet_grass | rain)`.

# %%
diag = InfluenceDiagram()
diag.add_node(
    ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
    )
)
diag.add_node(
    ChanceNode(
        name="wet_grass",
        parents=("rain",),
        states=("dry", "wet"),
        dist=lambda rain: dist.Categorical(
            probs=jnp.array([[0.80, 0.20], [0.05, 0.95]])[rain]
        ),
    )
)
print(diag.validate())

# %% [markdown]
# ## The prior
#
# `infer()` returns one typed result per query variable: a `Marginal`
# (probability vector) for discrete variables. `exact=True` means the result
# is exact — variable elimination, no sampling.

# %%
result = infer(diag, ["rain", "wet_grass"])
for name, marginal in result.items():
    print(name, "->", [round(p, 3) for p in marginal.values], "exact:", marginal.exact)

# %% [markdown]
# ## Evidence
#
# Observing the grass is wet updates the belief about rain. The engine
# conditions on `wet_grass = wet` (state index 1) and returns the exact
# posterior.

# %%
post = infer(diag, ["rain"], observed={"wet_grass": 1})
print("P(rain | wet_grass=wet) ->", [round(p, 3) for p in post["rain"].values])

# %% [markdown]
# ## Exact vs Monte-Carlo
#
# The auto-dispatch picked variable elimination (all-discrete diagram → exact).
# The same query through the NumPyro engine is a Monte-Carlo estimate: same
# posterior, drawn by enumeration with 2000 samples — close, but `exact=False`.

# %%
exact = infer(diag, ["rain"], observed={"wet_grass": 1}, engine="ve")
mc = infer(diag, ["rain"], observed={"wet_grass": 1}, engine="numpyro")
print(
    "ve      ->",
    [round(p, 3) for p in exact["rain"].values],
    "exact:",
    exact["rain"].exact,
)
print(
    "numpyro ->", [round(p, 3) for p in mc["rain"].values], "exact:", mc["rain"].exact
)
