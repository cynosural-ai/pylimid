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
# # Influence diagrams — interventional queries with policy=
#
# `solve()` told us what to do. Sometimes we want to ask "what would happen
# *if* we did X?" — without the diagram choosing for us. That is the
# `policy=` binding: clamp a decision to a fixed action and infer on the
# collapsed Bayesian network.
#
# Same medical model as the previous notebook.

# %%
import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import InferenceError, infer

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
# ## Decisions must be bound
#
# `infer()` refuses to run on an influence diagram with an unbound decision —
# "what is the posterior over recovery?" has no answer until you say what the
# treatment *is*.

# %%
try:
    infer(diag, ["recovery"])
except InferenceError as e:
    print("InferenceError:", e)

# %% [markdown]
# ## The counterfactual
#
# Bind the decision to each action in turn and compare the recovery
# distributions: `P(recovery | treat=no)` vs `P(recovery | treat=yes)`.

# %%
no_treatment = infer(diag, ["recovery"], policy={"treat": 0})
with_treatment = infer(diag, ["recovery"], policy={"treat": 1})
print(
    "P(recovery | treat=no)  ->", [round(p, 3) for p in no_treatment["recovery"].values]
)
print(
    "P(recovery | treat=yes) ->",
    [round(p, 3) for p in with_treatment["recovery"].values],
)

# %% [markdown]
# ## Conditioned on the disease
#
# The interesting version conditions on the patient's state: does treatment
# help the sick more than it hurts the healthy?

# %%
for disease_label, disease_state in [("healthy", 0), ("sick", 1)]:
    no = infer(
        diag, ["recovery"], observed={"disease": disease_state}, policy={"treat": 0}
    )
    yes = infer(
        diag, ["recovery"], observed={"disease": disease_state}, policy={"treat": 1}
    )
    print(
        f"{disease_label}: no treat -> {[round(p, 3) for p in no['recovery'].values]} | "
        f"treat -> {[round(p, 3) for p in yes['recovery'].values]}"
    )

# %% [markdown]
# ## Exact cross-check
#
# A bound decision behaves as ordinary evidence to the engine, so the same
# result is available through variable elimination directly with the decision
# in `observed` — confirming the `policy=` path computes the exact posterior.

# %%
from decisionpy.inference.ve import query

direct = query(
    diag.snapshot(),
    variables=["recovery"],
    observed={"disease": 1, "treat": 1},
)
print("ve with treat=1 in observed ->", [round(p, 3) for p in direct["recovery"]])

# %% [markdown]
# ## The full loop
#
# 1. `solve()` finds the optimal policy.
# 2. `infer(..., policy=)` evaluates what would happen under any policy —
#    including the optimal one.

# %%
from decisionpy.inference import solve

solution = solve(diag)
print("optimal policy:", solution.policy)
print("expected utility:", round(solution.expected_utility, 3))
