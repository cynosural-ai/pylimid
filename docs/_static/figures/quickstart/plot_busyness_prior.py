"""
Plot the LogNormal prior for busyness shown in the quickstart.

Run from the repository root with
``uv run python docs/_static/figures/quickstart/plot_busyness_prior.py``.
The parameters must match ``docs/getting_started/quickstart.md``.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpyro.distributions as dist

BUSYNESS_MEAN = 1.2
BUSYNESS_LOG_SD = 0.1


def main() -> None:
    """Plot the busyness prior and save it next to this script."""
    prior = dist.LogNormal(
        jnp.log(BUSYNESS_MEAN) - BUSYNESS_LOG_SD**2 / 2, BUSYNESS_LOG_SD
    )
    lower = BUSYNESS_MEAN * 0.8
    upper = BUSYNESS_MEAN * 1.2
    busyness = jnp.linspace(BUSYNESS_MEAN * 0.5, BUSYNESS_MEAN * 1.5, 800)
    density = jnp.exp(prior.log_prob(busyness))
    coverage = (prior.cdf(upper) - prior.cdf(lower)).item()

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.fill_between(
        busyness,
        density,
        where=(busyness >= lower) & (busyness <= upper),
        color="#A9C6E2",
        alpha=0.6,
        label=f"{coverage:.0%} between {lower:g} and {upper:g}",
    )
    ax.plot(
        busyness,
        density,
        color="#0A1F3D",
        linewidth=2.2,
        label=f"LogNormal prior (log-scale SD = {BUSYNESS_LOG_SD})",
    )
    ax.axvline(
        prior.mean, color="#f4a261", linestyle=":", label=f"Mean = {BUSYNESS_MEAN:g}"
    )
    ax.set(
        title="Prior belief about seasonal busyness",
        xlabel="Busyness multiplier",
        ylabel="Probability density",
        xlim=(BUSYNESS_MEAN * 0.5, BUSYNESS_MEAN * 1.5),
        ylim=(0, None),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(Path(__file__).with_name("busyness_prior.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
