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
# # Inference mechanics — the smallest Bayesian network
#
# A Bayesian network is an influence diagram with only chance nodes. Once the
# diagram is built (see `01_model_creation/workspace_and_validation.ipynb`),
# inference is one call: `infer()` runs the NumPyro engine — exact discrete
# enumeration for all-discrete diagrams, NUTS otherwise.
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
# (probability vector) for discrete variables. For an all-discrete diagram the
# NumPyro engine enumerates the discrete space, so the estimates are very
# close to the true probabilities.

# %%
result = infer(diag, ["rain", "wet_grass"])
for name, marginal in result.items():
    print(name, "->", [round(p, 3) for p in marginal.values])

# %% [markdown]
# ## Evidence
#
# Observing the grass is wet updates the belief about rain. The engine
# conditions on `wet_grass = wet` (state index 1) and returns the posterior:
# P(rain=yes | wet) = 0.2·0.95 / 0.43 ≈ 0.542.

# %%
post = infer(diag, ["rain"], observed={"wet_grass": 1})
print("P(rain | wet_grass=wet) ->", [round(p, 3) for p in post["rain"].values])

# %% [markdown]
# ## Continuous variables: raw draws
#
# A continuous variable has no per-state probability mass, so its result is
# the raw posterior draws. Adding a continuous node switches the engine from
# enumeration to NUTS.

# %%
diag.add_node(ChanceNode(name="temp", dist=lambda: dist.Normal(loc=20.0, scale=3.0)))
draws = infer(diag, ["temp"])
print("temp draws:", draws["temp"].values[:5], "...")
print("mean ≈", round(sum(draws["temp"].values) / len(draws["temp"].values), 2))
