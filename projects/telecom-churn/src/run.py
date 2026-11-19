"""Acquire UCI Iranian Churn data and generate aggregate evaluation reports."""

import argparse
import csv
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import acquire_data, features_and_target, profile_groups
from .figures import write_figures
from .model import run_modeling

PROJECT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
BINARY_METRICS = ("roc_auc", "average_precision", "brier", "precision", "recall", "f1")


def explore(raw, features, target, split, train_positions):
    """Describe data quality and training feature patterns using aggregate counts."""
    train = features.iloc[train_positions]
    numeric = train.select_dtypes(include="number").describe().transpose()
    numeric = numeric.round(6).astype(object).where(numeric.notna(), None)
    complaint_summary = (
        train.assign(churned=target.iloc[train_positions])
        .groupby("complains", observed=True)["churned"]
        .agg(customers="size", churners="sum", churn_rate="mean")
    )
    groups = profile_groups(raw)
    sizes = raw.groupby(groups)["churn"].agg(size="size", label_count="nunique")
    return {
        "rows": len(raw),
        "columns": len(raw.columns),
        "churners": int(target.sum()),
        "churn_rate": float(target.mean()),
        "blank_counts": {name: int(count) for name, count in raw.isna().sum().items()},
        "exact_duplicate_rows_after_first": int(raw.duplicated().sum()),
        "matching_profiles": len(sizes),
        "repeated_profile_groups": int(sizes["size"].gt(1).sum()),
        "mixed_label_profile_groups": int(sizes["label_count"].gt(1).sum()),
        "zero_voice_calls": int(features["no_voice_calls"].sum()),
        "charge_amount_range": [int(raw["charge_amount"].min()), int(raw["charge_amount"].max())],
        "train_numeric_summary": numeric.to_dict(orient="index"),
        "train_complaint_summary": complaint_summary.round(6).to_dict(orient="index"),
        "splits": split,
    }


def write_model_table(metrics, path):
    """Export validation and holdout metrics at the common cutoff."""
    with path.open("w", encoding="utf-8", newline="") as output:
        columns = ("model", "split", "threshold", *BINARY_METRICS)
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, model in metrics["models"].items():
            for split in ("validation", "test"):
                result = model[split]
                writer.writerow(
                    {"model": name, "split": split, **{key: result[key] for key in columns[2:]}}
                )


def run(raw_dir: Path, reports_dir: Path) -> None:
    """Train and evaluate the group-aware pipeline, then write aggregate outputs."""
    raw, provenance = acquire_data(raw_dir)
    features, target = features_and_target(raw)
    LOGGER.info("Loaded %s customer rows and %s eligible features", len(raw), len(features.columns))
    metrics, scores = run_modeling(features, target)
    metrics["source"] = provenance
    eda = explore(raw, features, target, metrics["split"], scores["train"]["positions"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".building-", dir=reports_dir) as temp:
        stage = Path(temp)
        for name, result in (("eda.json", eda), ("metrics.json", metrics)):
            (stage / name).write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        write_model_table(metrics, stage / "model-comparison.csv")
        write_figures(raw, metrics, scores, stage / "figures")
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                destination = reports_dir / file.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                file.replace(destination)

    LOGGER.info("Selected %s on validation average precision", metrics["selection"]["model"])
    LOGGER.info("Wrote metrics, EDA, and figures to %s", reports_dir)


def main():
    """Run the study with overridable download and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
