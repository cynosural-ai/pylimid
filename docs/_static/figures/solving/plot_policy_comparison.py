"""
Compare the fitted policies from the Solving guide.

Run with ``uv run python plot_policy_comparison.py`` from this directory.
The models and fitting settings come from the page's executable code cells.
"""

import re
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from pylimid import CategoricalPolicy, ContinuousPolicy


def main() -> None:
    """Execute the Solving guide and save its policy comparison figure."""
    guide = Path(__file__).resolve().parents[3] / "user_guide/solving.md"
    namespace: dict[str, Any] = {}
    for source in re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", guide.read_text(), re.S
    ):
        exec(compile(source, str(guide), "exec"), namespace)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        gridspec_kw={"width_ratios": (1, 2.4)},
        sharex="col",
        sharey="col",
        layout="constrained",
    )
    quantities = jnp.linspace(0.0, namespace["CAPACITY"], 1601)
    for (promotion_ax, order_ax), (name, title) in zip(
        axes,
        (
            ("initial_solution", "Initial fit: scale 0.25 kg"),
            ("solution", "New fit: scale 1 kg"),
        ),
        strict=True,
    ):
        solution = namespace[name]
        promotion = cast(CategoricalPolicy, solution.policy["promotion"])
        order = cast(ContinuousPolicy, solution.policy["order"])
        probabilities = promotion.probabilities().tolist()
        bars = promotion_ax.bar(
            promotion.states, probabilities, color=("#A9C6E2", "#0A1F3D"), width=0.6
        )
        for bar, probability in zip(bars, probabilities, strict=True):
            promotion_ax.text(
                bar.get_x() + bar.get_width() / 2,
                probability + 0.03,
                f"{probability:.2%}",
                ha="center",
            )
        promotion_ax.set(title=title, ylabel="Probability", ylim=(0, 1.15))
        promotion_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        promotion_ax.tick_params(labelbottom=True)

        preferred_promotion = int(promotion.probabilities().argmax())
        for weather, (label, color) in enumerate(
            (("Cloudy", "#8d99ae"), ("Sunny", "#f4a261"))
        ):
            distribution = order.distribution(
                weather=jnp.array(weather), promotion=jnp.array(preferred_promotion)
            )
            density = jnp.exp(distribution.log_prob(quantities))
            order_ax.plot(
                quantities,
                density,
                color=color,
                label=f"{label}: mean {distribution.mean:.1f} kg",
            )
            order_ax.fill_between(quantities, density, color=color, alpha=0.2)
        situation = "skipped" if preferred_promotion == 0 else "running"
        order_ax.set(
            title=f"Order with promotion {situation}",
            ylabel="Density",
            xlim=(0, namespace["CAPACITY"]),
            ylim=(0, None),
        )
        order_ax.legend(frameon=False)
        for ax in (promotion_ax, order_ax):
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
    axes[1, 1].set_xlabel("Ice cream ordered (kg)")
    fig.suptitle("Fitted policies before and after changing the order scale")
    output = Path(__file__).with_name("policy_comparison.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
