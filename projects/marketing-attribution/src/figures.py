"""Render campaign credit comparisons from the exported attribution aggregates."""

from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

SOURCE = "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/"
LICENSE = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
MODELS = (
    ("first_touch", "First touch", "#236492", -0.23),
    ("last_touch", "Last touch", "#176e6c", 0),
    ("linear_touch", "Linear", "#915823", 0.23),
)


def write_figures(campaigns: list[dict], metrics: dict, output_dir: Path,
                  filename: str = "campaign-credit.svg") -> None:
    """Compare credits for the campaigns with the highest linear credit."""
    ranked = sorted(campaigns, key=lambda row: (-row["linear_touch"], row["campaign"]))[:7]
    conversions = metrics["paths"]["qualified_conversions"]
    lookback = metrics["lookback_days"]
    scope = metrics.get("scope_label", "Source-linked")
    output_dir.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "svg.hashsalt": "marketing-attribution-campaign-credit",
    }):
        figure, axis = plt.subplots(figsize=(5.6, 4.65))
        try:
            figure.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.76)
            figure.text(0.04, 0.94, "Conversion credit by campaign", fontsize=16, weight="bold")
            figure.text(
                0.04, 0.885,
                f"{scope} · {conversions:,} paths · {lookback}-day window",
                fontsize=10, color=MUTED,
            )
            for key, label, color, offset in MODELS:
                axis.barh(
                    [position + offset for position in range(len(ranked))],
                    [row[key] for row in ranked],
                    height=0.20, color=color, label=label,
                )

            axis.set_yticks(range(len(ranked)), [f"#{row['campaign']}" for row in ranked])
            axis.set_ylim(max(1, len(ranked)) - 0.53, -0.53)
            axis.set_xlim(left=0)
            axis.set_xlabel("Attributed conversion credits", labelpad=9)
            axis.grid(axis="x", color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            axis.tick_params(length=0, pad=5)
            axis.spines[["left", "top", "right"]].set_visible(False)
            axis.spines["bottom"].set_color(GRID)
            axis.legend(
                loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3,
                frameon=False, borderaxespad=0, columnspacing=1.1,
            )
            footer = figure.text(
                0.04, 0.035, "Criteo AI Lab (2017) · CC BY-NC-SA 4.0",
                fontsize=9, color=MUTED,
            )
            footer.set_url(LICENSE)
            with StringIO() as buffer:
                figure.savefig(buffer, format="svg", metadata={
                    "Title": "First-touch, last-touch and linear campaign credits",
                    "Description": (
                        f"Top campaigns by linear credit. {scope} credits assigned to impression paths "
                        "from Criteo Attribution Modeling for Bidding Dataset."
                    ),
                    "Source": SOURCE,
                    "Rights": f"Criteo-derived chart licensed under CC BY-NC-SA 4.0 ({LICENSE}).",
                    "Date": None,
                })
                svg = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
            (output_dir / filename).write_text(svg, encoding="utf-8")
        finally:
            plt.close(figure)
