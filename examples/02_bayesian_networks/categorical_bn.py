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
# # Categorical Bayesian networks — the Asia-style network
#
# A 4-node network: `smoking` and `pollution` both cause `cancer`, and cancer
# shows up on an `xray`. This adds what the two-node example cannot show:
# multiple parents, multiple queries, and the classic "explaining away" effect.

# %%
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid.graph import ChanceNode, InfluenceDiagram
from pylimid.inference import infer

# %% [markdown]
# ## The model
#
# The cancer node has two parents, so its table is indexed with a single
# indexing operation over both (`T[smoking, pollution]`), one row per parent
# combination, each row of length `len(states)`.

# %%
diag = InfluenceDiagram()
diag.add_node(
    ChanceNode(
        name="smoking",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.7, 0.3])),
    )
)
diag.add_node(
    ChanceNode(
        name="pollution",
        states=("low", "high"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.9, 0.1])),
    )
)
diag.add_node(
    ChanceNode(
        name="cancer",
        parents=("smoking", "pollution"),
        states=("absent", "present"),
        dist=lambda smoking, pollution: dist.Categorical(
            probs=jnp.array(
                [
                    # smoking=no
                    [[0.99, 0.01], [0.40, 0.60]],  # pollution low, high
                    # smoking=yes
                    [[0.90, 0.10], [0.50, 0.50]],  # pollution low, high
                ]
            )[smoking, pollution]
        ),
    )
)
diag.add_node(
    ChanceNode(
        name="xray",
        parents=("cancer",),
        states=("neg", "pos"),
        dist=lambda cancer: dist.Categorical(
            probs=jnp.array([[0.80, 0.20], [0.10, 0.90]])[cancer]
        ),
    )
)
print(diag.validate())

# %% [markdown]
# ## Multiple queries
#
# One call, every variable at once.

# %%
prior = infer(diag, ["smoking", "pollution", "cancer", "xray"])
for name, posterior in prior.items():
    print(name, "->", [round(p, 3) for p in posterior.marginal()])

# %% [markdown]
# ## Evidence
#
# A positive xray makes cancer more likely, which in turn changes the odds on
# its causes.

# %%
post = infer(diag, ["smoking", "pollution", "cancer"], observed={"xray": 1})
for name, posterior in post.items():
    print(name, "->", [round(p, 3) for p in posterior.marginal()])

# %% [markdown]
# ## Explaining away
#
# Given the positive xray, learning that pollution was high *lowers* the
# posterior on smoking: pollution alone is a strong enough cause of cancer
# that it "explains away" the need for smoking as the explanation.
# P(smoking=yes | xray=pos) = 0.34 drops to
# P(smoking=yes | xray=pos, pollution=high) = 0.275.

# %%
post_with_pollution = infer(diag, ["smoking"], observed={"xray": 1, "pollution": 1})
print(
    "P(smoking | xray=pos)                  ->",
    [round(p, 3) for p in post["smoking"].marginal()],
)
print(
    "P(smoking | xray=pos, pollution=high)  ->",
    [round(p, 3) for p in post_with_pollution["smoking"].marginal()],
)

# %% [markdown]
# # The posterior against the exact answer
#
# The NumPyro engine enumerates the discrete space, so these are estimates
# of the exact posterior. Building the same network in pyAgrum — whose
# inference is exact — shows the agreement: P(cancer=present | xray=pos)
# ≈ 0.309 exactly, and the estimate lands within a percent.

# %%
import pyagrum as gum

bn = gum.BayesNet()
for name, states in [
    ("smoking", ["no", "yes"]),
    ("pollution", ["low", "high"]),
    ("cancer", ["absent", "present"]),
    ("xray", ["neg", "pos"]),
]:
    bn.add(gum.LabelizedVariable(name, name, states))
for parent, child in [
    ("smoking", "cancer"),
    ("pollution", "cancer"),
    ("cancer", "xray"),
]:
    bn.addArc(bn.idFromName(parent), bn.idFromName(child))
bn.cpt("smoking")[{}] = [0.7, 0.3]
bn.cpt("pollution")[{}] = [0.9, 0.1]
bn.cpt("cancer")[{"smoking": 0, "pollution": 0}] = [0.99, 0.01]
bn.cpt("cancer")[{"smoking": 0, "pollution": 1}] = [0.40, 0.60]
bn.cpt("cancer")[{"smoking": 1, "pollution": 0}] = [0.90, 0.10]
bn.cpt("cancer")[{"smoking": 1, "pollution": 1}] = [0.50, 0.50]
bn.cpt("xray")[{"cancer": 0}] = [0.80, 0.20]
bn.cpt("xray")[{"cancer": 1}] = [0.10, 0.90]

exact = gum.LazyPropagation(bn)
exact.setEvidence({"xray": "pos"})
exact.makeInference()

post_cancer = infer(diag, ["cancer"], observed={"xray": 1})
print(
    "numpyro (2000 samples) ->",
    [round(p, 3) for p in post_cancer["cancer"].marginal()],
)
print(
    "pyagrum (exact)        ->",
    [round(p, 3) for p in exact.posterior("cancer").tolist()],
)
