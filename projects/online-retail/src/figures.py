"""Render the exported monthly, customer, and cohort marts as static SVGs."""

from io import StringIO
from pathlib import Path
from textwrap import fill

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter

matplotlib.use("Agg")

INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
BLUE = "#236492"
TEAL = "#176e6c"
MISSING = "#e3e6ea"


def _axes(height=4.1, grid="y"):
    fig, ax = plt.subplots(figsize=(5.35, height), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(GRID)
    ax.tick_params(length=0, pad=7)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, output_dir, filename, title):
    try:
        with StringIO() as buffer:
            fig.savefig(buffer, format="svg", metadata={"Title": title, "Date": None})
            svg = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
        (output_dir / filename).write_text(svg, encoding="utf-8")
    finally:
        plt.close(fig)


def _pounds_tick(value, _):
    sign = "−" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 1_000_000:
        amount = f"{absolute / 1_000_000:g}m"
    elif absolute >= 1_000:
        amount = f"{absolute / 1_000:g}k"
    else:
        amount = f"{absolute:g}"
    return f"{sign}£{amount}"


def _plot_monthly_values(monthly):
    monthly = monthly.sort_values("month")
    dates = pd.to_datetime(monthly["month"]).tolist()
    complete = ~monthly["is_partial"].to_numpy(dtype=bool)
    positions = np.arange(len(monthly))
    fig, ax = _axes()
    for position in positions[~complete]:
        ax.axvspan(position - 0.46, position + 0.46, color="#ecf2f6", zorder=0)

    series = (
        ("gross_gbp", "Gross sales lines", BLUE, "-", "o"),
        ("net_gbp", "Net after credits", TEAL, "--", "D"),
    )
    for key, label, color, linestyle, marker in series:
        values = monthly[key].to_numpy(dtype=float)
        ax.plot(positions, np.where(complete, values, np.nan), label=label, color=color,
                linestyle=linestyle, marker=marker, markersize=4.5, linewidth=2)
        ax.plot(positions[~complete], values[~complete], color=color, linestyle="none",
                marker=marker, markerfacecolor="white", markersize=5, markeredgewidth=1.5)

    tick_positions = sorted({*positions[::2], *positions[~complete], len(monthly) - 1})
    tick_labels = []
    for position in tick_positions:
        month = dates[position]
        if not complete[position]:
            tick_labels.append(month.strftime("%b %Y\npartial"))
        elif position == 0 or position == len(monthly) - 1:
            tick_labels.append(month.strftime("%b\n%Y"))
        else:
            tick_labels.append(month.strftime("%b"))
    ax.set_xticks(tick_positions, tick_labels)
    ax.set(xlim=(-0.5, len(monthly) - 0.5), ylabel="Reported line value")
    if ax.get_ylim()[0] > 0:
        ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_pounds_tick))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, frameon=False)
    return fig


def _plot_segments(segments, monthly):
    segments = segments.sort_values(["customer_count", "segment"], ascending=[True, False])
    positions = np.arange(len(segments))
    counts = segments["customer_count"].tolist()
    fig, ax = _axes(grid="x")
    bars = ax.barh(positions, counts, height=0.53, color=BLUE)
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=5)
    ax.set_yticks(positions, [fill(str(name), width=20) for name in segments["segment"]])
    ax.set(xlim=(0, max(1, *counts) * 1.25), xlabel="Customers with known IDs")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    complete_months = monthly.loc[~monthly["is_partial"], "observed_through"]
    if not complete_months.empty:
        cutoff = pd.Timestamp(complete_months.max())
        ax.set_title(f"Snapshot through {cutoff:%d %b %Y}", loc="left", fontsize=12, color=MUTED, pad=10)
    return fig


def _label_color(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return "white" if luminance < 0.179 else "black"


def _plot_cohort_repeat(cohorts):
    cohorts = cohorts.loc[cohorts["month_offset"].between(0, 6)]
    matrix = (
        cohorts.pivot(index="cohort_month", columns="month_offset", values="repeat_share")
        .sort_index()
        .reindex(columns=range(7))
    )
    fig, ax = _axes(height=max(3.4, 1.45 + len(matrix) * 0.35), grid=None)
    if matrix.empty:
        ax.text(0.5, 0.5, "No complete cohort periods", ha="center", va="center")
        ax.set_axis_off()
        return fig

    colors = LinearSegmentedColormap.from_list("retail_blues", ("#f3f8fb", "#a8cadc", "#214e77"))
    colors = colors.with_extremes(bad=MISSING)
    shares = matrix.to_numpy(dtype=float)
    ax.imshow(shares, cmap=colors, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row, col in np.ndindex(shares.shape):
        share = shares[row, col]
        if np.isfinite(share):
            ax.text(col, row, f"{share:.0%}", ha="center", va="center", fontsize=10.5,
                    color=_label_color(colors(share)))

    ax.set_xticks(range(7), [str(offset) for offset in range(7)])
    ax.set_yticks(range(len(matrix)), pd.to_datetime(matrix.index).strftime("%Y-%m"))
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    ax.set(xlabel="Months since first observed purchase", ylabel="First observed month")
    ax.legend(handles=[Patch(color=MISSING, label="Complete period unavailable")],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    return fig


def write_figures(marts: dict, output_dir: Path) -> None:
    """Export deterministic figures from observed UK retailer transactions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "online-retail-figures",
        "svg.fonttype": "none",
    }):
        _save(_plot_monthly_values(marts["monthly"]), output_dir, "monthly-line-values.svg", "Monthly gross and net line values in GBP")
        _save(_plot_segments(marts["segments"], marts["monthly"]), output_dir, "rfm-segments.svg", "Customers by RFM segment")
        _save(_plot_cohort_repeat(marts["cohorts"]), output_dir, "cohort-repeat.svg", "Observed purchase activity by customer cohort")
