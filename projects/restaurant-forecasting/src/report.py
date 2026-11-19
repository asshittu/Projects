"""Export only aggregate forecast metrics and charts for review."""

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naïve",
    "hist_gradient_boosting": "Gradient boosting",
}


def plot_fold_errors(folds, metric, ylabel, destination):
    """Chart the same aggregate error metric for both models on each window."""
    positions = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
    try:
        for offset, (model, label) in enumerate(MODEL_LABELS.items()):
            values = [fold["models"][model][metric] for fold in folds]
            ax.bar(
                positions + (offset - 0.5) * 0.34,
                values,
                width=0.34,
                label=label,
                color=("#527b9c", "#176d69")[offset],
            )
        ax.set_xticks(positions, [fold["name"].replace("_", " ").title() for fold in folds])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        with StringIO() as svg, matplotlib.rc_context({"svg.hashsalt": "restaurant-forecasting"}):
            fig.savefig(svg, format="svg", metadata={"Title": ylabel, "Date": None})
            destination.write_text(
                "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n",
                encoding="utf-8",
            )
    finally:
        plt.close(fig)


def write_report(results, output_dir: Path) -> Path:
    """Write a small archive of aggregate results, with an explicit file allowlist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "restaurant-forecasting-results.zip"
    with TemporaryDirectory(dir=output_dir) as temp:
        stage = Path(temp)
        (stage / "metrics.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = ["metrics.json"]
        for metric, ylabel in (("mae", "Mean absolute error (visitors)"), ("rmsle", "RMSLE")):
            filename = f"{metric}-by-window.svg"
            plot_fold_errors(results["folds"], metric, ylabel, stage / filename)
            files.append(filename)
        temporary_archive = stage / archive_path.name
        with ZipFile(temporary_archive, "w", ZIP_DEFLATED) as archive:
            for filename in files:
                archive.writestr(ZipInfo(filename), (stage / filename).read_bytes(), compress_type=ZIP_DEFLATED)
        temporary_archive.replace(archive_path)
    return archive_path
