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
# # Model creation — the mutable workspace
#
# pylimid models an influence diagram as a **mutable workspace**: nodes and
# edges are added, edited, and removed freely by whatever drives the model (a
# script, a UI, an LLM), and inference is gated behind an explicit validation
# checkpoint. This notebook builds a small diagram piece by piece, shows the
# consistency gate in action, and renders the result live with Mermaid.

# %%
import jax.numpy as jnp
import numpyro.distributions as dist
from IPython.display import Markdown

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

# %% [markdown]
# ## The three node types
#
# - **Chance node** — a random variable. Its `dist` answers "given my parents'
#   values, what distribution do I have?". `states` declares the discrete
#   outcomes; omit it for a continuous variable.
# - **Decision node** — a variable the agent controls; `states` are its
#   actions.
# - **Utility node** — a deterministic payoff `values(parents) -> float`.

# %%
rain = ChanceNode(
    name="rain",
    states=("no", "yes"),
    dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
)
print(rain.name, rain.states)  # "rain" ("no", "yes")

# %%
print(rain.parents)  # no parents: a root
print(rain.consistency)  # CONSISTENT: dist set and parents match (none)

# %% [markdown]
# ## The consistency gate
#
# A node is allowed to be incomplete while you edit it. Its `consistency`
# state is computed on demand from the live fields:
#
# | state | meaning |
# | --- | --- |
# | `UNCONFIGURED` | the configurable field (`dist` / `values` / action space) is unset |
# | `STALE` | the field's signature does not name every parent |
# | `CONSISTENT` | ready for inference |

# %%
wet_grass = ChanceNode(name="wet_grass", parents=("rain",))
wet_grass.consistency  # UNCONFIGURED: no dist yet

# %%
# A **kwargs-only dist cannot prove it uses its parents, so it is STALE.
wet_grass.dist = lambda **kwargs: dist.Categorical(probs=jnp.array([0.2, 0.8]))
print(wet_grass.consistency)
print(wet_grass.consistency_message(wet_grass.consistency))

# %%
# Naming the parent fixes it: the gate can now verify coverage.
wet_grass.dist = lambda rain: dist.Categorical(
    probs=jnp.array([[0.80, 0.20], [0.05, 0.95]])[rain]
)
wet_grass.consistency

# %% [markdown]
# ## Assembling the diagram
#
# The `InfluenceDiagram` is the container. Adding an incomplete node is fine
# — `validate()` reports everything that blocks inference.

# %%
diag = InfluenceDiagram()
diag.add_node(rain)
diag.add_node(ChanceNode(name="wet_grass", parents=("rain",)))  # no dist yet

diag.validate()

# %%
diag["wet_grass"].dist = lambda rain: dist.Categorical(
    probs=jnp.array([[0.80, 0.20], [0.05, 0.95]])[rain]
)
diag.validate()  # clean

# %% [markdown]
# `snapshot()` is the checkpoint inference runs against: it validates and
# freezes a topologically ordered view.

# %%
snap = diag.snapshot()
snap.order

# %% [markdown]
# ## Live Mermaid rendering
#
# JupyterLab (≥ 4.1) renders Mermaid code blocks in markdown cells, so a
# diagram can be displayed inline — and re-rendered after every edit.


# %%
def render(diagram: InfluenceDiagram) -> Markdown:
    """Render a diagram as a Mermaid figure in the notebook."""
    return Markdown(f"```mermaid\n{diagram.to_mermaid()}\n```")


# %%
umbrella = InfluenceDiagram()
umbrella.add_node(
    ChanceNode(
        name="rain",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.8, 0.2])),
    )
)
umbrella.add_node(
    DecisionNode(name="umbrella", parents=("rain",), states=("no", "yes"))
)
umbrella.add_node(
    ChanceNode(
        name="wet",
        parents=("rain", "umbrella"),
        states=("dry", "wet"),
        dist=lambda rain, umbrella: dist.Categorical(
            probs=jnp.array(
                [
                    [[0.90, 0.10], [0.70, 0.30]],  # rain=0
                    [[0.40, 0.60], [0.05, 0.95]],  # rain=1
                ]
            )[rain, umbrella]
        ),
    )
)
umbrella.add_node(
    UtilityNode(
        name="dryness",
        parents=("rain", "umbrella"),
        values=lambda rain, umbrella: float(rain == umbrella),
    )
)
render(umbrella)

# %% [markdown]
# ## The workspace reacts to edits
#
# Remove the decision: its edge is scrubbed, `wet` loses a parent its `dist`
# still expects — so the gate marks it STALE, and the rendering shows it.

# %%
umbrella.remove_node("umbrella")
render(umbrella)

# %%
wet = umbrella["wet"]
print(wet.parents)
print(wet.consistency)

# %% [markdown]
# ## Next
#
# Inference over these diagrams: `examples/02_bayesian_networks/` and
# `examples/03_influence_diagrams/`.
