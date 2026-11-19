"""Acquire the UCI data and generate aggregate analytical reports."""

import argparse
import csv
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import acquire_data, features_and_target
from .figures import write_figures
from .model import run_modeling

PROJECT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
BINARY_METRICS = ("roc_auc", "average_precision", "brier", "precision", "recall", "f1")


def explore(raw, features, target, split):
    """Summarize completeness, target drift, and training feature distributions."""
    stop = split["train"]["stop"]
    train = features.iloc[:stop]
    train_target = target.iloc[:stop]
    numeric = train.select_dtypes(include="number").describe().transpose()
    numeric = numeric.round(6).astype(object).where(numeric.notna(), None)

    previous_outcome = (
        train.assign(subscribed=train_target)
        .groupby("poutcome", observed=True)["subscribed"]
        .agg(calls="size", subscriptions="sum", subscription_rate="mean")
    )

    return {
        "rows": len(raw),
        "columns": len(raw.columns),
        "subscriptions": int(target.sum()),
        "subscription_rate": float(target.mean()),
        "blank_counts": {key: int(value) for key, value in raw.isna().sum().items()},
        "unknown_counts": {
            name: int(raw[name].eq("unknown").sum())
            for name in ("job", "education", "contact", "poutcome")
        },
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "repeated_feature_profiles_after_first_row": int(features.duplicated().sum()),
        "no_previous_contact_rows": int(raw["pdays"].eq(-1).sum()),
        "negative_balance_rows": int(raw["balance"].lt(0).sum()),
        "multiple_current_campaign_contact_rows": int(raw["campaign"].gt(1).sum()),
        "train_numeric_summary": numeric.to_dict(orient="index"),
        "train_previous_outcome_summary": previous_outcome.round(6).to_dict(orient="index"),
        "splits": split,
    }


def write_model_table(metrics, path):
    """Export model comparisons with one row per model and evaluation split."""
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
    """Train and evaluate once, then export only aggregate results and figures."""
    raw, provenance = acquire_data(raw_dir)
    features, target = features_and_target(raw)
    LOGGER.info("Loaded %s rows with %s pre-contact features", len(raw), len(features.columns))

    metrics, scores = run_modeling(features, target)
    metrics["source"] = provenance
    eda = explore(raw, features, target, metrics["split"])
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
    """Run the reproducible analysis with overridable cache and report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
