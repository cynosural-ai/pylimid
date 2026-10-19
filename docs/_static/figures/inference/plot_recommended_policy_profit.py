"""Generate the figures from the inference guide's examples."""

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
    min(posterior.values.min().item() for posterior, _, _ in figures),
    max(posterior.values.max().item() for posterior, _, _ in figures),
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

policies = [
    (namespace["recommended_profit"], "Recommended policy", "#4477aa"),
    (namespace["alternative_profit"], "Always order 160", "#bb5500"),
]
comparison_bins = np.linspace(
    min(posterior.values.min().item() for posterior, _, _ in policies),
    max(posterior.values.max().item() for posterior, _, _ in policies),
    71,
)
fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
for posterior, label, color in policies:
    profits = np.asarray(posterior.values)
    ax.hist(
        profits,
        bins=comparison_bins,
        weights=np.full(profits.size, 100 / profits.size),
        color=color,
        alpha=0.5,
        label=f"{label} (mean ${posterior.mean():,.0f})",
    )
    ax.axvline(posterior.mean(), color=color, linestyle="--", linewidth=1.8)
ax.set(
    xlabel="Profit per day ($)",
    ylabel="Simulated days (%)",
    title="Recommended policy versus ordering 160 every day",
    xlim=(comparison_bins[0], comparison_bins[-1]),
)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.25)
ax.legend(frameon=False)
fig.savefig(Path(__file__).with_name("policy_comparison.png"), dpi=180)
plt.close(fig)

demand = np.asarray(namespace["sunny_demand"].values)
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
    150,
    color="#bb5500",
    linestyle="--",
    linewidth=1.8,
    label="Available stock: 150 ice creams",
)
ax.set(
    xlabel="Daily demand (ice creams)",
    ylabel="Simulated days (%)",
    title="Demand on a sunny day",
    xlim=(0, demand.max()),
)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig(Path(__file__).with_name("sunny_demand.png"), dpi=180)
plt.close(fig)

print(f"Alternative policy: mean ${namespace['alternative_profit'].mean():.2f}")
print(f"Sunny demand: mean {namespace['sunny_demand'].mean():.2f}")
print(f"Stockout probability: {namespace['stockout_probability']:.1%}")
