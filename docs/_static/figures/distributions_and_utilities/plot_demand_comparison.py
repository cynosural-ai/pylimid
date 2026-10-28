"""
Plot the demand assumptions from the Distributions and utilities guide.

Run with ``uv run python plot_demand_comparison.py`` from this directory.
The distributions and parameters come from the page's executable code cells.
"""

import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    """Execute the guide and save its demand comparison figure."""
    guide = (
        Path(__file__).resolve().parents[3]
        / "user_guide/distributions_and_utilities.md"
    )
    namespace: dict[str, Any] = {}
    for source in re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", guide.read_text(), re.S
    ):
        exec(compile(source, str(guide), "exec"), namespace)

    observations = {
        "weather": 1,
        "promotion": 1,
        "busyness": namespace["BUSYNESS_MEAN"],
        "bookings": 0,
    }
    distributions = (
        (namespace["demand_dist"](**observations), "Gamma", "#4477aa"),
        (
            namespace["nonnegative_demand"](**observations),
            "Truncated Gaussian",
            "#bb5500",
        ),
    )
    demand = jnp.linspace(0.0, 60.0, 800)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for distribution, label, color in distributions:
        density = jnp.exp(distribution.log_prob(demand))
        ax.plot(demand, density, label=label, color=color, linewidth=2.2)
    ax.set(
        xlabel="Demand (kg)",
        ylabel="Probability density",
        title=(
            "Sunny day, promotion running, normal hotel bookings\n"
            f"Busyness fixed at {observations['busyness']:g}"
        ),
        xlim=(0, 60),
        ylim=(0, None),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    output = Path(__file__).with_name("demand_comparison.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
