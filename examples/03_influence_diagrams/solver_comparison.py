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
# # Influence diagrams — solve() with the NumPyro intervention scan
#
# `solve()` runs the NumPyro intervention scan (Strategy B): it enumerates
# the discrete policy space and estimates each policy's expected utility by
# forward sampling, keeping the best. The expected utility is a Monte-Carlo
# estimate — the seed makes it reproducible, and mixed and continuous
# diagrams are handled by the same forward sampling.
#
# Model: does the patient have the disease? Do we treat? Does the patient
# recover? The payoff is 100 per recovery minus 20 if we treated.

# %%
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import solve
from decisionpy.inference.numpyro.solver import solve as numpyro_solve

# %% [markdown]
# ## The model (all categorical)
#
# `disease` is a chance node, `treat` is a decision (its parents are its
# *information set* — what the doctor observes when choosing), `recovery` is
# a chance node, and `utility` is the payoff.

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
# ## The optimal policy
#
# `solve()` returns a Solution: the optimal action per information-set
# assignment, and the expected utility under that policy — do not treat the
# healthy, do treat the sick. The exact value of the optimum is 78:
# `0.6 * 90 + 0.4 * 60`; the estimate lands close.

# %%
solution = solve(diag)
print("policy:", solution.policy["treat"])
print("expected utility:", round(solution.expected_utility, 3))

# %% [markdown]
# ## The estimate vs the exact answer
#
# The same model built in pyAgrum — whose LIMID solver is exact — shows
# where the remaining gap lives: the scan finds the same policy, and its
# expected utility wobbles around pyAgrum's exact 78. The difference is the
# Monte-Carlo noise, and nothing else.

# %%
import itertools
import math

import pyagrum as gum

g = gum.InfluenceDiagram()
g.addChanceNode(gum.LabelizedVariable("disease", "disease", ["healthy", "sick"]))
g.addDecisionNode(gum.LabelizedVariable("treat", "treat", ["no", "yes"]))
g.addChanceNode(gum.LabelizedVariable("recovery", "recovery", ["no", "yes"]))
g.addUtilityNode(gum.LabelizedVariable("utility", "utility", ["pay"]))
for parent, child in [
    ("disease", "treat"),
    ("disease", "recovery"),
    ("treat", "recovery"),
    ("recovery", "utility"),
    ("treat", "utility"),
]:
    g.addArc(g.idFromName(parent), g.idFromName(child))
g.cpt("disease")[{}] = [0.6, 0.4]
for disease, treat in itertools.product(range(2), range(2)):
    probs = {
        (0, 0): [0.10, 0.90],
        (0, 1): [0.08, 0.92],
        (1, 0): [0.60, 0.40],
        (1, 1): [0.20, 0.80],
    }[(disease, treat)]
    g.cpt("recovery")[{"disease": disease, "treat": treat}] = probs
for recovery, treat in itertools.product(range(2), range(2)):
    g.utility("utility")[{"recovery": recovery, "treat": treat}] = (
        100.0 * recovery - 20.0 * treat
    )

ie = gum.ShaferShenoyLIMIDInference(g)
ie.makeInference()
exact_meu = float(ie.MEU()["mean"])
exact_policy = {}
for disease in range(2):
    for action in range(2):
        if math.isclose(
            float(ie.optimalDecision("treat")[{"disease": disease, "treat": action}]),
            1.0,
            abs_tol=1e-6,
        ):
            exact_policy[(disease,)] = action

print(
    "numpyro (2000 samples):",
    round(solution.expected_utility, 3),
    solution.policy["treat"],
)
print("pyagrum (exact):       ", exact_meu, exact_policy)

# %% [markdown]
# ## The estimate wobbles with the seed
#
# The expected utility is a Monte-Carlo estimate, not the mathematical
# optimum. Run the scan with different seeds and it wobbles around 78 —
# close enough to choose between actions, but honest about being an
# estimate.

# %%
for seed in range(3):
    solution = numpyro_solve(diag.snapshot(), rng_key=jax.random.PRNGKey(seed))
    print(
        f"seed {seed}: EU = {solution.expected_utility:.3f}",
        f"policy = {solution.policy['treat']}",
    )

# %% [markdown]
# ## Continuous outcomes are no obstacle
#
# Make `recovery` a continuous health score: `health | disease, treat` is
# Gaussian. The intervention scan handles it unchanged — forward sampling
# does not care whether a chance node is discrete or continuous.

# %%
mixed = InfluenceDiagram()
mixed.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
mixed.add_node(DecisionNode(name="treat", parents=("disease",), states=("no", "yes")))
mixed.add_node(
    ChanceNode(
        name="health",
        parents=("disease", "treat"),
        dist=lambda disease, treat: dist.Normal(
            loc=jnp.array(
                [
                    [90.0, 92.0],  # healthy
                    [40.0, 80.0],  # sick
                ]
            )[disease, treat],
            scale=5.0,
        ),
    )
)
mixed.add_node(
    UtilityNode(
        name="utility",
        parents=("health", "treat"),
        values=lambda health, treat: float(health) - 20.0 * treat,
    )
)

mc = solve(mixed)
print("policy:", mc.policy["treat"], "EU:", round(mc.expected_utility, 2))

# %% [markdown]
# ## Same decision, same expected value
#
# The continuous version has the same structure as the categorical one:
# treating the healthy buys 2 points of health for 20 of cost, treating the
# sick buys 40 for 20. So the optimal policy is the same — treat only the
# sick — and the expected utility is again `0.6 * 90 + 0.4 * 60 = 78`.

# %%
print("0.6 * 90 + 0.4 * 60 =", 0.6 * 90 + 0.4 * 60)
