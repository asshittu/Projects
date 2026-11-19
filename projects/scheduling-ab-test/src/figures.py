"""Render scheduling simulation estimates and their confidence intervals."""

from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
BLUE = "#236492"
TEAL = "#176e6c"


def _axes(title: str):
    fig = Figure(figsize=(6.1, 2.85), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_title(title, loc="left", color=INK, fontsize=14, pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig, ax


def _interval(ax, y, estimate, interval, color):
    ax.errorbar(
        estimate, y,
        xerr=[[estimate - interval["lower"]], [interval["upper"] - estimate]],
        fmt="o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=0.8,
        elinewidth=2.1, capsize=5, capthick=1.6, zorder=3,
    )


def _publish_rate(outcome: dict) -> Figure:
    fig, ax = _axes("SIMULATED · First schedule published")
    limits = []
    for y, key, color in ((1, "control", BLUE), (0, "guided", TEAL)):
        arm = outcome["arms"][key]
        estimate = arm["rate"]
        interval = arm["interval"]
        _interval(ax, y, estimate, interval, color)
        ax.annotate(
            f"{estimate:.1%}", (estimate, y), xytext=(0, 12), textcoords="offset points",
            ha="center", va="bottom", color=INK,
        )
        limits.extend((interval["lower"], interval["upper"]))

    lower, upper = min(limits), max(limits)
    padding = max(upper - lower, 0.01) * 0.20
    ax.set(
        xlabel="Seven-day publish rate · Wilson 95% CI",
        xlim=(max(0.0, lower - padding), min(1.0, upper + padding)),
        ylim=(-0.47, 1.55),
    )
    ax.set_yticks([1, 0], ["Standard editor", "Guided publishing"])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    return fig


def _support_guardrail(outcome: dict) -> Figure:
    fig, ax = _axes("SIMULATED · Support request guardrail")
    contrast = outcome["contrast"]
    difference = 100 * contrast["difference"]
    interval = {side: 100 * value for side, value in contrast["interval"].items()}
    threshold = 100 * outcome["threshold"]

    ax.axvline(0, color=MUTED, linestyle=":", linewidth=1.6, label="No difference")
    ax.axvline(
        threshold, color=TEAL, linestyle="--", linewidth=1.6,
        label=f"Tolerance {threshold:+.1f} pp",
    )
    _interval(ax, 0, difference, interval, BLUE)
    ax.annotate(
        f"{difference:+.2f} pp", (difference, 0), xytext=(0, 13), textcoords="offset points",
        ha="center", va="bottom", color=INK,
    )

    lower, upper = min(interval["lower"], threshold, 0), max(interval["upper"], threshold, 0)
    padding = max(upper - lower, 0.05) * 0.17
    ax.set(
        xlabel="Guided minus standard (pp) · Newcombe 95% CI",
        xlim=(lower - padding, upper + padding), ylim=(-0.22, 0.45),
    )
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncols=2, frameon=False)
    return fig


def write_figures(metrics: dict, output_dir: Path) -> None:
    """Save compact WebP charts from the synthetic experiment's aggregate metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "legend.fontsize": 10.5,
    }):
        outcomes = metrics["outcomes"]
        for outcome, draw, filename in (
            ("publish", _publish_rate, "publish-rate.webp"),
            ("support", _support_guardrail, "support-guardrail.webp"),
        ):
            draw(outcomes[outcome]).savefig(output_dir / filename, dpi=180, pil_kwargs={"quality": 88})
