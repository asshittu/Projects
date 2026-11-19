"""Write static SVG figures from the original data and evaluated models."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")

INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
BLUE = "#236492"
SLATE = "#64788d"
TEAL = "#176e6c"
PARTITIONS = ("train", "validation", "test")
MODEL_STYLES = {
    "dummy": (SLATE, "--", "o"),
    "logistic_regression": (BLUE, "-", "s"),
    "random_forest": (TEAL, "-.", "D"),
}


def _axes():
    fig, ax = plt.subplots(figsize=(5.35, 4.1), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(GRID)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
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


def _plot_cohort_prevalence(raw, metrics):
    names, rates, counts = [], [], []
    for partition in PARTITIONS:
        cohort = metrics["split"][partition]
        target = raw["y"].iloc[cohort["start"]:cohort["stop"]]
        names.append({"train": "Training", "validation": "Validation", "test": "Holdout"}[partition])
        rates.append(target.eq("yes").mean())
        counts.append(len(target))

    fig, ax = _axes()
    bars = ax.bar(names, rates, color=("#537c9c", "#34729a", "#214e77"), width=0.55)
    ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=5, color=INK)
    ax.set_xticks(range(len(names)), [f"{name}\n{count:,} rows" for name, count in zip(names, counts)])
    ax.set_ylim(0, min(1.08, max(rates) * 1.22))
    ax.set_xlabel("Earlier to later in source order", labelpad=12)
    ax.set_ylabel("Observed subscription rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    return fig


def _plot_precision_recall(metrics, test_scores):
    fig, ax = _axes()
    for key, (color, linestyle, _) in MODEL_STYLES.items():
        model = metrics["models"][key]
        precision, recall, _ = precision_recall_curve(test_scores["y_true"], test_scores["probabilities"][key])
        ax.step(
            recall, precision, where="post", color=color, linestyle=linestyle, linewidth=2,
            label=f"{model['label']} · AP {model['test']['average_precision']:.3f}",
        )

    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    return fig


def _plot_calibration(metrics):
    fig, ax = _axes()
    ax.plot((0, 1), (0, 1), color="#778796", linewidth=1.2, linestyle=":", label="Perfect calibration")
    for key, (color, linestyle, marker) in MODEL_STYLES.items():
        model = metrics["models"][key]
        bins = [bin_ for bin_ in model["test"]["calibration"] if bin_["count"]]
        marker_face = "none" if key == "dummy" else color
        ax.plot(
            [bin_["mean_probability"] for bin_ in bins],
            [bin_["observed_rate"] for bin_ in bins],
            color=color, linestyle=linestyle, marker=marker, markerfacecolor=marker_face,
            markersize=6 if key == "dummy" else 4, linewidth=1.7,
            label=model["label"],
        )
        for bin_ in bins:
            if bin_["count"] < 30:
                ax.annotate(
                    f"n={bin_['count']}", (bin_["mean_probability"], bin_["observed_rate"]),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, color=MUTED,
                )

    ax.set(xlabel="Mean predicted probability", ylabel="Observed subscription rate", xlim=(0, 1), ylim=(0, 1.04))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=2, frameon=False)
    return fig


def _plot_contact_budget(metrics):
    scenarios = metrics["policy"]["test"]["capacity_scenarios"]
    scenarios = sorted(scenarios, key=lambda scenario: scenario["capacity"])
    ranking = [scenario["conversions"] for scenario in scenarios]
    random = [scenario["expected_random_conversions"] for scenario in scenarios]
    positions = np.arange(len(scenarios))
    width = 0.32

    fig, ax = _axes()
    ranked_bars = ax.bar(positions - width / 2, ranking, width, color=BLUE, label="Model ranking")
    random_bars = ax.bar(positions + width / 2, random, width, color="#a9bfd0", label="Random expectation")
    ax.bar_label(ranked_bars, labels=[f"{count:,}" for count in ranking], padding=4, color=INK)
    ax.bar_label(random_bars, labels=[f"{count:,.1f}" for count in random], padding=4, color=INK)
    ax.set_xticks(positions, [f"{scenario['capacity']:.0%}" for scenario in scenarios])
    ax.set_ylim(0, max(1, *ranking, *random) * 1.20)
    ax.set(xlabel="Share of holdout rows selected", ylabel="Subscriptions among selected rows")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=2, frameon=False)
    return fig


def _plot_confusion_matrix(metrics):
    counts = np.asarray(metrics["policy"]["test"]["fixed_threshold"]["confusion_matrix"])

    fig, ax = _axes()
    ax.grid(False)
    map_ = matplotlib.colors.LinearSegmentedColormap.from_list("bank_blues", ("#edf3f7", "#8ab3ce"))
    ax.imshow(counts, cmap=map_, vmin=0, vmax=max(1, counts.max()), aspect="auto")
    ax.spines[["bottom", "left"]].set_visible(False)
    ax.set_xticks((0, 1), ["No subscription", "Subscription"])
    ax.set_yticks((0, 1), ["No subscription", "Subscription"])
    ax.set(xlabel="Predicted outcome", ylabel="Actual outcome")
    labels = (("True negatives", "False positives"), ("False negatives", "True positives"))
    for row, col in np.ndindex(counts.shape):
        count = counts[row, col]
        ax.text(col, row - 0.10, f"{count:,}", ha="center", va="center", fontsize=18, weight="bold")
        ax.text(col, row + 0.15, labels[row][col], ha="center", va="center", fontsize=11)
    return fig


def write_figures(raw: pd.DataFrame, metrics: dict, scores: dict, output_dir: Path) -> None:
    """Export five deterministic figures using the holdout and original row order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_scores = scores["test"]
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "bank-marketing-figures",
        "svg.fonttype": "none",
    }):
        _save(_plot_cohort_prevalence(raw, metrics), output_dir, "cohort-prevalence.svg", "Subscription rate by ordered split")
        _save(_plot_precision_recall(metrics, test_scores), output_dir, "precision-recall.svg", "Holdout precision and recall")
        _save(_plot_calibration(metrics), output_dir, "calibration.svg", "Holdout probability calibration")
        _save(_plot_contact_budget(metrics), output_dir, "contact-budget.svg", "Subscriptions captured at fixed contact budgets")
        _save(_plot_confusion_matrix(metrics), output_dir, "confusion-matrix.svg", "Holdout confusion matrix at the validation-selected cutoff")
