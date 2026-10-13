"""Generate the profit figures from the inference guide's examples."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("guide", type=Path, help="Markdown guide containing the examples.")
args = parser.parse_args()

namespace = {}
for source in re.findall(
    r"```\{code-cell\} ipython3\n(.*?)\n```", args.guide.read_text(), re.S
):
    exec(compile(source, str(args.guide), "exec"), namespace)

figures = [
    (
        namespace["recommended_profit"],
        "Daily profit under the recommended policy",
        "recommended_policy_profit.png",
    ),
    (
        namespace["sunny_profit"],
        "Profit on a sunny day with high hotel bookings",
        "sunny_policy_profit.png",
    ),
]
bins = np.linspace(
    min(min(posterior.values) for posterior, _, _ in figures),
    max(max(posterior.values) for posterior, _, _ in figures),
    71,
)
for posterior, title, filename in figures:
    profits = np.asarray(posterior.values)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    ax.hist(
        profits,
        bins=bins,
        weights=np.full(profits.size, 100 / profits.size),
        color="#4477aa",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        posterior.mean(),
        color="#bb5500",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean profit: ${posterior.mean():,.0f}",
    )
    ax.set(
        xlabel="Profit per day ($)",
        ylabel="Simulated days (%)",
        title=title,
        xlim=(bins[0], bins[-1]),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    output = Path(__file__).with_name(filename)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"{title}: mean ${posterior.mean():.2f}; {profits.size} draws")
    print(f"Saved {output}")
