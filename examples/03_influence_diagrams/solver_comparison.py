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
# # Influence diagrams — the intervention scan vs the exact answer
#
# This notebook uses the intervention scan (`solve(method="scan")`): it
# enumerates the discrete policy space and estimates each policy's expected
# utility by forward sampling, keeping the best — every policy evaluated in
# one vmapped pass. The expected utility is a Monte-Carlo estimate — the
# seed makes it reproducible, and mixed and continuous diagrams are handled
# by the same forward sampling.
#
# Model: does the patient have the disease? Do we treat? Does the patient
# recover? The payoff is 100 per recovery minus 20 if we treated.

# %%
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import solve
from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

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
solution = solve(diag, method="scan")
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
    solution = solve(diag, method="scan", rng_key=jax.random.PRNGKey(seed))
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
        values=lambda health, treat: health - 20.0 * treat,
    )
)

mc = solve(mixed, method="scan")
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

# %% [markdown]
# ## The Oil Wildcatter — pyAgrum exact vs the scan
#
# Up to now the reference answers were computed by hand on small models. For
# a bigger, classic problem we let pyAgrum solve the diagram *exactly* (LIMID
# with the Shafer-Shenoy algorithm) and compare its answer against the scan.
#
# The Oil Wildcatter is the textbook example of a decision under uncertainty
# (Raiffa's classic, and the model shipped with pyAgrum's tutorials). An oil
# deposit may be Dry, Wet, or Soaking. We may run an expensive test whose
# report — closed, open, or diffuse — is informative about the deposit;
# then we decide whether to drill. Drilling pays off per deposit type, the
# test costs 10. Six nodes, two decisions, and the drilling decision
# observes the test report.
#
# The exact optimum: run the test, then drill unless the report is
# "diffuse". MEU = 22.5.

# %%
from _oil_wildcatter import pyagrum_solution, pylimid_diagram

oil = pylimid_diagram()
print(oil.validate())

meu, exact_policy = pyagrum_solution()
print("pyAgrum MEU:", meu)
print("pyAgrum Testing:", exact_policy["Testing"])
reports = {
    r: exact_policy["Drilling"][(1, i)]
    for i, r in enumerate(("closed", "open", "diffuse"))
}
print("pyAgrum Drilling (report -> action):", reports)

# %% [markdown]
# ## The scan agrees on the realized policy
#
# The scan evaluates every policy by forward sampling — with 128 candidate
# policies (the drilling rule has 2 actions over its 6-assignment
# information set) the solver runs the whole enumeration in one
# vmapped pass, so this cell uses the default 2000 samples and still
# finishes in well under a second.

# %%
oil_solution = solve(oil, method="scan")
print("scan MEU:", round(oil_solution.expected_utility, 3))

# Drilling's info set is (Testing, TestResult); the branch that matters is
# the one the optimal policy realizes — the test runs (Testing=Yes).
reports = {
    r: oil_solution.policy["Drilling"][(1, i)]
    for i, r in enumerate(("closed", "open", "diffuse"))
}
print("scan Drilling given the test ran:", reports)

# %% [markdown]
# ## Reading the comparison
#
# - **MEU**: the exact answer is 22.5; the scan's estimate lands within its
#   Monte-Carlo noise (the reward spreads over −70..200, so a 2–3 point
#   wobble is expected even at 2000 samples).
# - **The decision rule**: restricted to the branch that actually occurs —
#   the test is run, so the drilling rule conditions on the report — both
#   engines agree: drill on closed and open, not on diffuse.
# - **The no-test branch**: the scan fills in a rule for `Testing=No` rows
#   too, but that branch is never realized under the optimal policy (its
#   prior probability under the policy is zero), so those rows do not
#   affect the expected utility. pyAgrum's `optimalDecision` drops the
#   decision parent from the returned table entirely.
