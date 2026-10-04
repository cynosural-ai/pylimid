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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear-Gaussian Bayesian networks
#
# An all-continuous network: every node is Gaussian and its mean is a linear
# function of its parents' values,
#
#     x | parents ~ Normal(loc = a + sum_i b_i * parent_i, scale = sigma)
#
# with every parent continuous too. A 3-node chain: `x1` is a unit Gaussian,
# `x2` depends on `x1`, and `x3` on `x2`. Unlike the categorical examples,
# there is no probability table — the `dist` callable just builds a `Normal`
# whose `loc` is a linear expression in the parents. The NumPyro engine
# samples the posterior with NUTS.

# %%
import numpyro.distributions as dist

from pylimid.graph import ChanceNode, InfluenceDiagram
from pylimid.inference import infer

# %% [markdown]
# ## The model

# %%
diag = InfluenceDiagram()
diag.add_node(ChanceNode(name="x1", dist=lambda: dist.Normal(loc=0.0, scale=1.0)))
diag.add_node(
    ChanceNode(
        name="x2",
        parents=("x1",),
        dist=lambda x1: dist.Normal(loc=2.0 + 1.5 * x1, scale=0.5),
    )
)
diag.add_node(
    ChanceNode(
        name="x3",
        parents=("x2",),
        dist=lambda x2: dist.Normal(loc=-1.0 - 0.7 * x2, scale=0.3),
    )
)
print(diag.validate())

# %% [markdown]
# ## The priors
#
# The NumPyro engine runs NUTS on the continuous model: the posterior of
# every query variable is the raw draws. The analytic marginals are
# x2 ~ N(2, 2.5) and x3 ~ N(-2.4, 1.315) — the empirical moments of the
# draws land close.

# %%
import jax.numpy as jnp

prior = infer(diag, ["x1", "x2", "x3"])
for name, d in prior.items():
    draws = jnp.array(d.values)
    print(name, "->", f"mean={float(draws.mean()):.4f}, var={float(draws.var()):.4f}")

# %% [markdown]
# ## Evidence
#
# Observing `x3 = 0.4` sharpens the posterior on its ancestors. The exact
# answer is x1 | x3=0.4 ~ N(-2.2357, 0.1616) — the evidence pushes x1 down
# because it must explain a small x3. The draws follow it.

# %%
post = infer(diag, ["x1", "x2"], observed={"x3": 0.4})
for name, d in post.items():
    draws = jnp.array(d.values)
    print(name, "->", f"mean={float(draws.mean()):.4f}, var={float(draws.var()):.4f}")
