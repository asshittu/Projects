"""Export static figures for the churn data audit and model evaluation."""

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
TEAL = "#176e6c"
MODEL_STYLES = {
    "dummy": ("#64788d", "--"),
    "logistic_regression": (BLUE, "-"),
    "random_forest": (TEAL, "-."),
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


def _plot_churn_rates(groups, xlabel):
    """Plot observed rates with each group's event count and denominator."""
    rates = [positives / count if count else 0 for _, positives, count in groups]
    labels = [f"{name}\n{positives:,} of {count:,}" if count else f"{name}\nNo records"
              for name, positives, count in groups]
    fig, ax = _axes()
    positions = np.arange(len(groups))
    bars = ax.bar(positions, rates, color=("#527f9b", "#2b7192", "#214e77")[:len(groups)], width=0.55)
    ax.bar_label(bars, labels=[f"{rate:.1%}" if group[2] else "" for rate, group in zip(rates, groups)],
                 padding=5, color=INK)
    ax.set(xticks=positions, xticklabels=labels, xlabel=xlabel, ylabel="Observed churn rate")
    ax.set_ylim(0, min(1.08, max(0.1, *rates) * 1.22))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    return fig


def _plot_split_prevalence(metrics):
    groups = []
    for key, name in (("train", "Training"), ("validation", "Validation"), ("test", "Holdout")):
        split = metrics["split"][key]
        groups.append((name, split["positives"], split["n"]))
    return _plot_churn_rates(groups, "Data partition")


def _plot_complaint_churn(raw, train_scores):
    train = raw.iloc[train_scores["positions"]]
    groups = []
    for value, name in ((0, "No complaint"), (1, "Complaint recorded")):
        outcomes = train.loc[train["complains"].eq(value), "churn"]
        groups.append((name, int(outcomes.sum()), len(outcomes)))
    return _plot_churn_rates(groups, "Complaint status in training records")


def _plot_precision_recall(metrics, test_scores):
    fig, ax = _axes()
    for key, (color, linestyle) in MODEL_STYLES.items():
        model = metrics["models"][key]
        precision, recall, _ = precision_recall_curve(test_scores["y_true"], test_scores["probabilities"][key])
        ax.step(
            recall, precision, where="post", color=color, linestyle=linestyle, linewidth=2,
            label=f"{model['label']} · AP {model['test']['average_precision']:.3f}",
        )
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    return fig


def _plot_retention_budget(metrics):
    scenarios = metrics["policy"]["test"]["capacity_scenarios"]
    scenarios = sorted(scenarios, key=lambda scenario: scenario["capacity"])
    ranking = [scenario["churners"] for scenario in scenarios]
    random = [scenario["expected_random_churners"] for scenario in scenarios]
    positions = np.arange(len(scenarios))
    width = 0.32

    fig, ax = _axes()
    ranked_bars = ax.bar(positions - width / 2, ranking, width, color=BLUE, label="Model ranking")
    random_bars = ax.bar(positions + width / 2, random, width, color="#a9bfd0", label="Random expectation")
    ax.bar_label(ranked_bars, labels=[f"{count:,}" for count in ranking], padding=4, color=INK)
    ax.bar_label(random_bars, labels=[f"{count:,.1f}" for count in random], padding=4, color=INK)
    ax.set_xticks(positions, [f"{scenario['capacity']:.0%}" for scenario in scenarios])
    ax.set_ylim(0, max(1, *ranking, *random) * 1.20)
    ax.set(xlabel="Share of holdout selected", ylabel="Observed churners selected")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=2, frameon=False)
    return fig


def _plot_confusion_matrix(metrics):
    counts = np.asarray(metrics["policy"]["test"]["fixed_threshold"]["confusion_matrix"])
    fig, ax = _axes()
    ax.grid(False)
    palette = matplotlib.colors.LinearSegmentedColormap.from_list("churn_blues", ("#edf3f7", "#8ab3ce"))
    ax.imshow(counts, cmap=palette, vmin=0, vmax=max(1, counts.max()), aspect="auto")
    ax.spines[["bottom", "left"]].set_visible(False)
    ax.set_xticks((0, 1), ["Not flagged", "Flagged"])
    ax.set_yticks((0, 1), ["No churn", "Churned"])
    ax.set(xlabel="Model decision at fixed cutoff", ylabel="Observed outcome")
    labels = (("True negatives", "False positives"), ("False negatives", "True positives"))
    for row, col in np.ndindex(counts.shape):
        ax.text(col, row - 0.10, f"{counts[row, col]:,}", ha="center", va="center", fontsize=18, weight="bold")
        ax.text(col, row + 0.15, labels[row][col], ha="center", va="center", fontsize=11)
    return fig


def write_figures(raw: pd.DataFrame, metrics: dict, scores: dict, output_dir: Path) -> None:
    """Write deterministic SVGs from aggregate results and the evaluated predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "telecom-churn-figures",
        "svg.fonttype": "none",
    }):
        _save(_plot_split_prevalence(metrics), output_dir, "split-prevalence.svg", "Churn rate by data partition")
        _save(_plot_complaint_churn(raw, scores["train"]), output_dir, "complaint-churn.svg", "Training churn rates by recorded complaint status")
        _save(_plot_precision_recall(metrics, scores["test"]), output_dir, "precision-recall.svg", "Holdout precision and recall for churn prediction")
        _save(_plot_confusion_matrix(metrics), output_dir, "confusion-matrix.svg", "Holdout confusion matrix at the validation-selected cutoff")
        _save(_plot_retention_budget(metrics), output_dir, "retention-budget.svg", "Historical churners flagged at fixed selection budgets")
