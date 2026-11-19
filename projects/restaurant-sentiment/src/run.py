"""Download the UCI Yelp subset and export held-out sentiment results."""

import argparse
import csv
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import acquire_data
from .figures import write_figures
from .model import run_modeling

PROJECT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def explore(data, train_positions):
    """Describe training-only review lengths and class coverage."""
    train = data.iloc[train_positions]
    word_counts = train["text"].str.split().str.len()
    distributions = {}
    for label in (0, 1):
        lengths = word_counts.loc[train["label"].eq(label)]
        distributions[str(label)] = {
            "rows": len(lengths),
            "median_words": float(lengths.median()),
            "minimum_words": int(lengths.min()),
            "maximum_words": int(lengths.max()),
        }
    return {
        "train_word_counts_by_label": distributions,
        "sentences": len(data),
        "blank_counts": {name: int(count) for name, count in data.isna().sum().items()},
    }


def write_model_table(metrics, path):
    """Write split-level metrics in an aggregate comparison table."""
    with path.open("w", encoding="utf-8", newline="") as output:
        fields = ("model", "split", "macro_f1", "accuracy", "average_precision")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, model in metrics["models"].items():
            for split in ("validation", "test"):
                result = model[split]
                writer.writerow({"model": name, "split": split, **{key: result[key] for key in fields[2:]}})


def run(raw_dir: Path, reports_dir: Path) -> None:
    """Run the frozen evaluation and export scoring summaries and examples."""
    data, provenance = acquire_data(raw_dir)
    LOGGER.info("Loaded %s unique labeled Yelp sentences", len(data))
    metrics, details = run_modeling(data)
    metrics["source"] = provenance
    eda = explore(data, details["train"]["positions"])
    excerpts = {
        "source": "UCI Sentiment Labelled Sentences, Yelp subset, CC BY 4.0",
        "selection": "Highest positive scores among false positives and lowest positive scores among false negatives",
        "examples": {
            kind: [
                {key: entry[key] for key in ("excerpt", "positive_probability", "true_label", "predicted_label")}
                for entry in entries
            ]
            for kind, entries in details["errors"].items()
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".building-", dir=reports_dir) as temp:
        stage = Path(temp)
        for name, result in (("metrics.json", metrics), ("eda.json", eda), ("error-examples.json", excerpts)):
            (stage / name).write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        write_model_table(metrics, stage / "model-comparison.csv")
        write_figures(metrics, stage / "figures")
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                destination = reports_dir / file.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                file.replace(destination)

    LOGGER.info("Selected %s by validation macro F1", metrics["selection"]["model"])
    LOGGER.info("Wrote reports and figures to %s", reports_dir)


def main():
    """Run with optional cache and report directory overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
