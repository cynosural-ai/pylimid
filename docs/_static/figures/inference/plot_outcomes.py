"""
Generate the figures from the Inference guide.

Run with ``uv run python plot_outcomes.py`` from this directory.
The models, policies and sampling settings come from the page's executable cells.
"""

import re
from pathlib import Path
from typing import Any

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Execute the Inference guide and save its four outcome figures."""
    guide = Path(__file__).resolve().parents[3] / "user_guide/inference.md"
    namespace: dict[str, Any] = {}
    for source in re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", guide.read_text(), re.S
    ):
        exec(compile(source, str(guide), "exec"), namespace)

    _plot_profit(
        namespace["profit_draws"],
        "Daily profit under the fitted policies",
        Path(__file__).with_name("fitted_policy_profit.png"),
    )
    _plot_comparison(
        (
            (namespace["sunny_profit"], "Original information", "#4477aa"),
            (namespace["report_profit"], "Hotel report", "#bb5500"),
        ),
        "Sunny days with high hotel bookings",
        Path(__file__).with_name("hotel_report_comparison.png"),
    )
    _plot_comparison(
        (
            (namespace["profit_draws"], "Fitted order policy", "#4477aa"),
            (namespace["alternative_profit"], "Always order 16 kg", "#bb5500"),
        ),
        "Fitted order policy versus ordering 16 kg every day",
        Path(__file__).with_name("policy_comparison.png"),
    )
    _plot_demand(
        namespace["sunny_demand"].values,
        Path(__file__).with_name("sunny_demand.png"),
    )


def _plot_profit(values: jax.Array, title: str, output: Path) -> None:
    """Draw a histogram of daily profit with its estimated mean."""
    profits = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    ax.hist(
        profits,
        bins=70,
        weights=np.full(profits.size, 100 / profits.size),
        color="#4477aa",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        profits.mean(),
        color="#bb5500",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean profit: ${profits.mean():,.0f}",
    )
    ax.set(xlabel="Total profit per day ($)", ylabel="Draws (%)", title=title)
    _finish_plot(fig, ax, output)


def _plot_comparison(
    series: tuple[tuple[jax.Array, str, str], ...], title: str, output: Path
) -> None:
    """Compare profit distributions using common bins and separate mean lines."""
    bins = np.linspace(
        min(float(values.min()) for values, _, _ in series),
        max(float(values.max()) for values, _, _ in series),
        71,
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for values, label, color in series:
        profits = np.asarray(values)
        ax.hist(
            profits,
            bins=bins,
            weights=np.full(profits.size, 100 / profits.size),
            color=color,
            alpha=0.45,
            label=f"{label} (mean ${profits.mean():,.0f})",
        )
        ax.axvline(profits.mean(), color=color, linestyle="--", linewidth=1.8)
    ax.set(xlabel="Total profit per day ($)", ylabel="Draws (%)", title=title)
    ax.margins(y=0.2)
    _finish_plot(fig, ax, output)


def _plot_demand(values: jax.Array, output: Path) -> None:
    """Draw sunny-day demand with the available stock marked."""
    demand = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    ax.hist(
        demand,
        bins=np.linspace(0, demand.max(), 71),
        weights=np.full(demand.size, 100 / demand.size),
        color="#4477aa",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        20.0,
        color="#bb5500",
        linestyle="--",
        linewidth=1.8,
        label="Available stock: 20 kg",
    )
    ax.set(
        xlabel="Demand (kg)",
        ylabel="Draws (%)",
        title="Demand on sunny days when the promotion runs",
        xlim=(0, demand.max()),
    )
    _finish_plot(fig, ax, output)


def _finish_plot(fig: Any, ax: Any, output: Path) -> None:
    """Apply the shared figure style and save the result."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
