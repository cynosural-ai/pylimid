"""
Plot the fitted order probabilities from the quickstart's two models.

Run from the repository root with
``uv run python docs/_static/figures/quickstart/plot_policy.py``.
The models and fitting settings come directly from the quickstart's code cells.
"""

import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from pylimid import CategoricalPolicy


def main() -> None:
    """Execute the quickstart examples and save both order-policy figures."""
    guide = Path(__file__).resolve().parents[3] / "getting_started/quickstart.md"
    namespace: dict[str, Any] = {}
    for source in re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", guide.read_text(), re.S
    ):
        exec(compile(source, str(guide), "exec"), namespace)

    for name, title, filename in (
        ("solution", "Fitted order policy", "order_policy.png"),
        (
            "prior_solution",
            "Fitted order policy with uncertain seasonal busyness",
            "prior_order_policy.png",
        ),
    ):
        _plot_policy(
            namespace[name].policy["order"],
            namespace["ORDERS"],
            title,
            Path(__file__).with_name(filename),
        )


def _plot_policy(
    policy: CategoricalPolicy,
    orders: jax.Array,
    title: str,
    output: Path,
) -> None:
    """Draw paired bars for the cloudy- and sunny-day action probabilities."""
    stock = np.asarray(orders)
    width = 7.5
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    for weather, (label, color, offset) in enumerate(
        (("Cloudy", "#8d99ae", -width / 2), ("Sunny", "#f4a261", width / 2))
    ):
        probabilities = np.asarray(policy.probabilities(weather=jnp.array(weather)))
        bars = ax.bar(
            stock + offset,
            100 * probabilities,
            width=width,
            color=color,
            label=label,
        )
        for bar, probability in zip(bars, probabilities, strict=True):
            if probability >= 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{probability:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    ax.set(
        title=title,
        xlabel="Ice creams stocked",
        ylabel="Probability of choosing order",
        xticks=stock,
        ylim=(0, 108),
        yticks=np.arange(0, 101, 20),
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
