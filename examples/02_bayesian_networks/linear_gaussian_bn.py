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
# # Linear-Gaussian Bayesian networks — exact Gaussian inference
#
# An all-continuous network: every node is Gaussian and its mean is a linear
# function of its parents' values,
#
#     x | parents ~ Normal(loc = a + sum_i b_i * parent_i, scale = sigma)
#
# with every parent continuous too. A 3-node chain: `x1` is a unit Gaussian,
# `x2` depends on `x1`, and `x3` on `x2`. Unlike the categorical examples,
# there is no probability table — the `dist` callable just builds a `Normal`
# whose `loc` is a linear expression in the parents.

# %%
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, InfluenceDiagram
from decisionpy.inference import infer

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
# ## Exact priors
#
# `engine="lg"` runs exact linear-Gaussian variable elimination: the
# posterior of every query variable is a `Gaussian(mean, variance)`, not raw
# draws. For comparison, the analytic marginals are
# x2 ~ N(2, 2.5) and x3 ~ N(-2.4, 1.315).

# %%
prior = infer(diag, ["x1", "x2", "x3"], engine="lg")
for name, g in prior.items():
    print(name, "->", f"N(mean={g.mean:.4f}, var={g.variance:.4f})")

# %% [markdown]
# ## Evidence
#
# Observing `x3 = 0.4` sharpens the posterior on its ancestors. The exact
# answer is x1 | x3=0.4 ~ N(-2.2357, 0.1616) — the evidence pushes x1 down
# because it must explain a small x3.

# %%
post = infer(diag, ["x1", "x2"], observed={"x3": 0.4}, engine="lg")
for name, g in post.items():
    print(name, "->", f"N(mean={g.mean:.4f}, var={g.variance:.4f})")

# %% [markdown]
# ## The engines agree
#
# Exact (linear-Gaussian elimination) and Monte-Carlo (NumPyro NUTS) on the
# same query. The draws are approximate, so their empirical mean and
# standard deviation should land close to the exact Gaussian.

# %%
import jax.numpy as jnp

exact = infer(diag, ["x1"], observed={"x3": 0.4}, engine="lg")
mc = infer(diag, ["x1"], observed={"x3": 0.4}, engine="numpyro")
draws = jnp.array(mc["x1"].values)
print(
    "exact   ->",
    f"N(mean={exact['x1'].mean:.4f}, var={exact['x1'].variance:.4f})",
)
print(
    "numpyro ->",
    f"mean={float(draws.mean()):.4f}, std={float(draws.std()):.4f} "
    f"(from {len(draws)} draws)",
)
