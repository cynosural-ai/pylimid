"""
Draw both fitted policies from the two-decision quickstart.

Run from the repository root with
``uv run python docs/_static/figures/quickstart/plot_policy.py``.
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

from pylimid import CategoricalPolicy, ContinuousPolicy, Solution


def main() -> None:
    """Execute the quickstart and save its two policy figures."""
    guide = Path(__file__).resolve().parents[3] / "getting_started/quickstart.md"
    namespace: dict[str, Any] = {}
    for source in re.findall(
        r"```\{code-cell\} ipython3\n(.*?)\n```", guide.read_text(), re.S
    ):
        exec(compile(source, str(guide), "exec"), namespace)

    for name, title, filename in (
        ("solution", "Fitted policies", "order_policy.png"),
        (
            "prior_solution",
            "Fitted policies with uncertain seasonal busyness",
            "prior_order_policy.png",
        ),
    ):
        _plot_policy(
            namespace[name],
            namespace["CAPACITY"],
            title,
            Path(__file__).with_name(filename),
        )


def _plot_policy(solution: Solution, capacity: float, title: str, output: Path) -> None:
    """Draw promotion probabilities beside weather-dependent order densities."""
    promotion = cast(CategoricalPolicy, solution.policy["promotion"])
    order = cast(ContinuousPolicy, solution.policy["order"])
    fig, (promotion_ax, order_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 3.8),
        width_ratios=(1, 2.4),
        layout="constrained",
    )
    probabilities = promotion.probabilities().tolist()
    bars = promotion_ax.bar(
        promotion.states, probabilities, color=("#A9C6E2", "#0A1F3D"), width=0.6
    )
    for bar, probability in zip(bars, probabilities, strict=True):
        promotion_ax.text(
            bar.get_x() + bar.get_width() / 2,
            probability + 0.025,
            f"{probability:.3%}",
            ha="center",
        )
    promotion_ax.set(title="Evening: promotion", ylabel="Probability", ylim=(0, 1.14))
    promotion_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    quantities = jnp.linspace(0.0, capacity, 1601)
    for weather, (label, color) in enumerate(
        (("Cloudy", "#8d99ae"), ("Sunny", "#f4a261"))
    ):
        distribution = order.distribution(
            weather=jnp.array(weather), promotion=jnp.array(1)
        )
        density = jnp.exp(distribution.log_prob(quantities))
        order_ax.plot(
            quantities,
            density,
            color=color,
            label=f"{label}: mean {distribution.mean:.1f} kg",
        )
        order_ax.fill_between(quantities, density, color=color, alpha=0.2)
    order_ax.set(
        title="Morning: order when the promotion runs",
        xlabel="Ice cream ordered (kg)",
        ylabel="Density",
        xlim=(0, capacity),
        ylim=(0, None),
    )
    order_ax.legend(frameon=False)
    for ax in (promotion_ax, order_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
