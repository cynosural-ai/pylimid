"""
Draw the demand distributions shown in the quickstart.

Run from the repository root with
``uv run python docs/_static/figures/quickstart/plot_demand.py``.
The numbers must match ``docs/getting_started/quickstart.md``.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpyro.distributions as dist

WEATHER_STATES = ("cloudy", "sunny")
DEMAND_MEANS = jnp.array([40.0, 120.0])
COLORS = ("#8d99ae", "#f4a261")


def demand_dist(weather):
    """Return the Gamma demand distribution for a weather index."""
    return dist.Gamma(concentration=4.0, rate=4.0 / DEMAND_MEANS[weather])


def main():
    """Plot both demand densities and save them next to this script."""
    demand = jnp.linspace(0.0, 300.0, 500)
    fig, ax = plt.subplots(figsize=(8, 4))
    for weather, (name, color) in enumerate(zip(WEATHER_STATES, COLORS, strict=True)):
        density = jnp.exp(demand_dist(weather).log_prob(demand))
        ax.plot(demand, density, color=color, label=f"{name} day")
        ax.axvline(DEMAND_MEANS[weather], color=color, linestyle=":")
    ax.set_xlabel("ice creams demanded")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("demand.png"), dpi=150)


if __name__ == "__main__":
    main()
