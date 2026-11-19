"""Render the observed assignment shares, conversion rates, and primary contrast."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

matplotlib.use("Agg")

INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
BLUE = "#236492"
TEAL = "#176e6c"
ARMS = (("ad", "Ad", BLUE), ("psa", "PSA", TEAL))
POSITIONS = (1, 0)


def _axes(height=3.1):
    fig, ax = plt.subplots(figsize=(5.35, height), layout="constrained")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, output_dir, filename, title):
    destination = output_dir / filename
    try:
        fig.savefig(destination, metadata={"Title": title, "Date": None})
    finally:
        plt.close(fig)
    lines = destination.read_text(encoding="utf-8").splitlines()
    destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _plot_allocation(groups):
    fig, ax = _axes()
    proportions = [groups[key]["observed_share"] for key, _, _ in ARMS]
    bars = ax.barh(POSITIONS, proportions, color=[color for _, _, color in ARMS], height=0.40)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in proportions], padding=5, color=INK)
    labels = [f"{label}\n{groups[key]['n']:,} users" for key, label, _ in ARMS]
    ax.set_yticks(POSITIONS, labels)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set(xlabel="Share of observed users", xlim=(0, 1.16), ylim=(-0.5, 1.5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    return fig


def _plot_interval(ax, y, value, lower, upper, color):
    ax.hlines(y, lower, upper, color=color, linewidth=2.2)
    ax.vlines((lower, upper), y - 0.05, y + 0.05, color=color, linewidth=1.6)
    ax.scatter(value, y, s=43, color=color, edgecolor="white", linewidth=0.65, zorder=3)


def _plot_conversion_rates(groups):
    fig, ax = _axes()
    edges = []
    for y, (key, _, color) in zip(POSITIONS, ARMS):
        group = groups[key]
        value = group["conversion_rate"]
        interval = group["wilson_95_ci"]
        lower, upper = interval["lower"], interval["upper"]
        _plot_interval(ax, y, value, lower, upper, color)
        ax.annotate(f"{value:.2%}", (value, y), xytext=(0, 13), textcoords="offset points", ha="center")
        edges.extend((lower, upper))

    ax.set_yticks(POSITIONS, [label for _, label, _ in ARMS])
    extent = max(max(edges) - min(edges), 0.002)
    ax.set_xlim(max(0, min(edges) - extent * 0.20), max(edges) + extent * 0.30)
    ax.set(ylim=(-0.45, 1.55), xlabel="Conversion rate with Wilson 95% interval")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig


def _plot_absolute_lift(contrast, hurdle):
    fig, ax = _axes(height=3.0)
    interval = contrast["newcombe_95_ci"]
    value = 100 * contrast["absolute_difference"]
    lower = 100 * interval["lower"]
    upper = 100 * interval["upper"]
    references = [0.0]

    ax.axvline(0, color=MUTED, linestyle=":", linewidth=1.35, label="No difference")
    if hurdle is not None:
        benchmark = 100 * hurdle["minimum_absolute_difference"]
        references.append(benchmark)
        ax.axvline(
            benchmark, color=TEAL, linestyle="--", linewidth=1.35,
            label=f"Illustrative hurdle ({benchmark:+.2f} pp)",
        )

    _plot_interval(ax, 0, value, lower, upper, BLUE)
    ax.annotate(f"{value:+.2f} pp", (value, 0), xytext=(0, 13), textcoords="offset points", ha="center")
    ax.annotate("Newcombe 95% interval", (value, 0), xytext=(0, -17), textcoords="offset points",
                ha="center", va="top", fontsize=10, color=MUTED)
    ax.set_yticks([0], ["Ad minus PSA"])
    left, right = min(lower, *references), max(upper, *references)
    extent = max(right - left, 0.05)
    ax.set_xlim(left - extent * 0.12, right + extent * 0.18)
    ax.set(ylim=(-0.28, 0.29), xlabel="Absolute lift (percentage points)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncols=2, frameon=False)
    return fig


def write_figures(results: dict, output_dir: Path) -> None:
    """Export deterministic SVGs using only measured aggregate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = results["groups"]
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "marketing-ab-test-figures",
        "svg.fonttype": "none",
    }):
        _save(_plot_allocation(groups), output_dir, "group-allocation.svg", "Observed group allocation")
        _save(_plot_conversion_rates(groups), output_dir, "conversion-rates.svg", "Conversion rates with Wilson 95% intervals")
        _save(
            _plot_absolute_lift(results["contrast"], results.get("practical_hurdle")),
            output_dir, "absolute-lift.svg", "Absolute lift with Newcombe 95% interval",
        )
