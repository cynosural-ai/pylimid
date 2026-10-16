"""
Plot the demand assumptions from the Distributions and utilities guide.

Run with ``uv run python plot_demand_comparison.py`` from this directory.
The parameters must match ``docs/user_guide/distributions_and_utilities.md``.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpyro.distributions as dist

DEMAND_MEANS = jnp.array([40.0, 120.0])
BOOKINGS_BOOST = jnp.array([1.0, 1.5])

# Sunny weather and normal hotel bookings, as in the figure caption.
mean = DEMAND_MEANS[1] * BOOKINGS_BOOST[0]
distributions = [
    (dist.Gamma(concentration=4.0, rate=4.0 / mean), "Gamma", "#4477aa"),
    (
        dist.TruncatedNormal(loc=mean, scale=mean / 2, low=0.0),
        "Truncated Gaussian",
        "#bb5500",
    ),
]

demand = jnp.linspace(0.0, 400.0, 800)
fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
for distribution, label, color in distributions:
    density = jnp.exp(distribution.log_prob(demand))
    ax.plot(demand, density, label=label, color=color, linewidth=2.2)

ax.set(
    xlabel="Demand (ice creams)",
    ylabel="Probability density",
    title="Demand on a sunny day with normal hotel bookings",
    xlim=(0, 400),
    ylim=(0, None),
)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig(Path(__file__).with_name("demand_comparison.png"), dpi=180)
plt.close(fig)
