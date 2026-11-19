"""Render measured holdout results as lightweight, deterministic SVGs."""

from io import StringIO
from pathlib import Path
from textwrap import fill

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter

matplotlib.use("Agg")

INK = "#162a41"
MUTED = "#475c70"
GRID = "#e1e8ed"
BLUE = "#236492"
TEAL = "#176e6c"
SLATE = "#637e92"
MODEL_ORDER = ("dummy", "tfidf_unigram", "tfidf_unigram_bigram")
CLASSES = ("negative", "positive")


def _axes(grid="y"):
    fig, ax = plt.subplots(figsize=(5.35, 3.9), layout="constrained")
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


def _plot_model_comparison(metrics):
    models = metrics["models"]
    selected = metrics["selection"]["model"]
    scores = [models[key]["test"]["macro_f1"] for key in MODEL_ORDER]
    labels = [fill(models[key]["label"], width=21) for key in MODEL_ORDER]
    selected_position = MODEL_ORDER.index(selected)
    labels[selected_position] += "\nSelected on validation"
    positions = np.arange(len(MODEL_ORDER))
    colors = [BLUE if key == selected else SLATE for key in MODEL_ORDER]

    fig, ax = _axes(grid="x")
    bars = ax.barh(positions, scores, color=colors, height=0.48)
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=5)
    ax.set_yticks(positions, labels)
    ax.get_yticklabels()[selected_position].set_color(INK)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set(xlim=(0, 1.13), xlabel="Holdout macro-F1")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    return fig


def _plot_confusion_matrix(evaluation):
    counts = np.asarray(evaluation["confusion_matrix"])
    fig, ax = _axes(grid=None)
    colors = LinearSegmentedColormap.from_list("sentiment_blues", ("#edf3f7", "#8ab3ce"))
    ax.imshow(counts, cmap=colors, vmin=0, vmax=max(1, counts.max()), aspect="auto", interpolation="nearest")
    ax.spines[["bottom", "left"]].set_visible(False)
    class_labels = [name.title() for name in CLASSES]
    ax.set_xticks((0, 1), class_labels)
    ax.set_yticks((0, 1), class_labels)
    ax.set(xlabel="Predicted sentiment", ylabel="Actual sentiment")
    labels = (("True negatives", "False positives"), ("False negatives", "True positives"))
    for row, col in np.ndindex(counts.shape):
        ax.text(col, row - 0.10, f"{counts[row, col]:,}", ha="center", va="center", fontsize=18, weight="bold")
        ax.text(col, row + 0.15, labels[row][col], ha="center", va="center", fontsize=11)
    return fig


def _plot_class_metrics(evaluation):
    scores = evaluation["class_metrics"]
    positions = np.arange(len(CLASSES))
    width = 0.24
    styles = (("precision", "Precision", BLUE), ("recall", "Recall", TEAL), ("f1", "F1", SLATE))
    fig, ax = _axes()
    for offset, (key, label, color) in zip((-width, 0, width), styles):
        values = [scores[name][key] for name in CLASSES]
        bars = ax.bar(positions + offset, values, width=width * 0.89, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10)

    ax.set_xlim(-0.48, len(CLASSES) - 0.52)
    ax.set_ylim(0, 1.11)
    ax.set_xticks(positions, [name.title() for name in CLASSES])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.set(xlabel="Sentiment class", ylabel="Holdout score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3, frameon=False)
    return fig


def write_figures(metrics: dict, output_dir: Path) -> None:
    """Export SVGs from untouched holdout scores for the three model candidates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = metrics["models"][metrics["selection"]["model"]]["test"]
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 10.5,
        "svg.hashsalt": "restaurant-sentiment-figures",
        "svg.fonttype": "none",
    }):
        _save(_plot_model_comparison(metrics), output_dir, "holdout-comparison.svg", "Holdout macro-F1 by model")
        _save(_plot_confusion_matrix(selected), output_dir, "confusion-matrix.svg", "Selected model holdout confusion matrix")
        _save(_plot_class_metrics(selected), output_dir, "class-metrics.svg", "Selected model holdout precision, recall, and F1 by sentiment")
