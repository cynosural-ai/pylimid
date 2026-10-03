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
# # Influence diagrams — solve()
#
# Add a decision and a utility to the chance nodes and the diagram stops being
# a Bayesian network: now the question is not "what is the posterior?" but
# "what should I do?". `solve()` answers that — the optimal policy and the
# expected utility it achieves.
#
# Model: does the patient have the disease? Do we treat? Does the patient
# recover? The payoff is 100 per recovery minus 20 if we treated.

# %%
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import solve
from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

# %% [markdown]
# ## The model
#
# `disease` is a chance node, `treat` is a decision (its parents are its
# *information set* — what the doctor observes when choosing), `recovery` is a
# chance node, and `utility` is a payoff.

# %%
diag = InfluenceDiagram()
diag.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
diag.add_node(DecisionNode(name="treat", parents=("disease",), states=("no", "yes")))
diag.add_node(
    ChanceNode(
        name="recovery",
        parents=("disease", "treat"),
        states=("no", "yes"),
        dist=lambda disease, treat: dist.Categorical(
            probs=jnp.array(
                [
                    [[0.10, 0.90], [0.08, 0.92]],  # healthy
                    [[0.60, 0.40], [0.20, 0.80]],  # sick
                ]
            )[disease, treat]
        ),
    )
)
diag.add_node(
    UtilityNode(
        name="utility",
        parents=("recovery", "treat"),
        values=lambda recovery, treat: 100.0 * recovery - 20.0 * treat,
    )
)
print(diag.validate())

# %% [markdown]
# ## Render it
#
# `diag` draws itself: chance nodes are circles, decisions rectangles,
# utilities diamonds. JupyterLab renders the SVG through `_repr_svg_`, which
# shells out to the Graphviz `dot` binary; `diag.to_mermaid()` stays
# available as text for LLMs and GitHub.

# %%
diag

# %% [markdown]
# ## Solve it
#
# `solve()` returns a Solution: the optimal action per information-set
# assignment, and the expected utility under that policy.

# %%
solution = solve(diag)
print("expected utility:", round(solution.expected_utility, 3))
print("policy:")
for decision, info in solution.policy.items():
    print(" ", decision, info)

# %% [markdown]
# ## Reading the policy
#
# - disease healthy: do **not** treat (expected 90 vs 92-20=72).
# - disease sick: **do** treat (60 vs 40).
#
# The expected utility is the prior-weighted average:
# `0.6 * 90 + 0.4 * 60 = 78`.

# %%
print("0.6 * 90 + 0.4 * 60 =", 0.6 * 90 + 0.4 * 60)
