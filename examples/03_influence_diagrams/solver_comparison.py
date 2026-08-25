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
# # Influence diagrams — two solvers for one model
#
# `solve()` has two engines behind it: bucket elimination (exact, the default
# for all-categorical diagrams) and the NumPyro intervention scan (a
# Monte-Carlo estimate, and the only one that handles mixed and continuous
# diagrams). Same model, same question — "what should I do?" — answered two
# ways.
#
# Model: does the patient have the disease? Do we treat? Does the patient
# recover? The payoff is 100 per recovery minus 20 if we treated.

# %%
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode
from decisionpy.inference import InferenceError, solve
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
# ## Both solvers agree on what to do
#
# Bucket elimination computes the exact optimum; the intervention scan
# enumerates the same policy space and estimates each policy's expected
# utility by forward sampling. Both land on the same policy — do not treat
# the healthy, do treat the sick — and the same expected utility, 78.

# %%
exact = solve(diag)
mc = solve(diag, engine="numpyro")

for label, solution in (("bucket elimination", exact), ("numpyro (2000 samples)", mc)):
    print(f"{label}:")
    print("  policy:", solution.policy["treat"])
    print("  expected utility:", round(solution.expected_utility, 3))
    print("  exact:", solution.exact)

# %% [markdown]
# ## The difference is the noise
#
# The `exact` flag tells the two apart: bucket elimination returns the
# mathematical optimum, the intervention scan a Monte-Carlo estimate of it.
# Run the scan a few times and the estimate wobbles around 78 — the seed is
# fixed by default, but each engine is honest about what it computed.

# %%
for seed in range(3):
    solution = numpyro_solve(diag.snapshot(), rng_key=jax.random.PRNGKey(seed))
    print(
        f"seed {seed}: EU = {solution.expected_utility:.3f}",
        f"policy = {solution.policy['treat']}",
    )

# %% [markdown]
# ## Where they part ways: continuous outcomes
#
# Now make `recovery` a continuous health score: `health | disease, treat`
# is Gaussian. The intervention scan handles it unchanged — forward
# sampling does not care whether a chance node is discrete or continuous.
# Bucket elimination refuses it loudly.

# %%
mixed = InfluenceDiagram()
mixed.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
mixed.add_node(
    DecisionNode(name="treat", parents=("disease",), states=("no", "yes"))
)
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

try:
    solve(mixed)
except InferenceError as e:
    print("bucket elimination:", e)

mc = solve(mixed, engine="numpyro")
print("numpyro:      policy:", mc.policy["treat"], "EU:", round(mc.expected_utility, 2))

# %% [markdown]
# ## Same decision, same expected value
#
# The continuous version has the same structure as the categorical one:
# treating the healthy buys 2 points of health for 20 of cost, treating the
# sick buys 40 for 20. So the optimal policy is the same — treat only the
# sick — and the expected utility is again `0.6 * 90 + 0.4 * 60 = 78`.
#
# The estimate lands close to 78; the remaining gap is the Monte-Carlo
# noise, exactly what `exact=False` announces.

# %%
print("0.6 * 90 + 0.4 * 60 =", 0.6 * 90 + 0.4 * 60)
