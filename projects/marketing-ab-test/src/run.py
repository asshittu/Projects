"""Download the verified Kaggle dataset and report aggregate A/B evidence."""

import argparse
import csv
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from .analysis import run_analysis
from .data import acquire_data
from .figures import write_figures

PROJECT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def explore(data):
    """Verify observed group coverage and describe exposure after group assignment."""
    exposure = (
        data.groupby("test_group", observed=True)["total_ads"]
        .agg(["min", "median", "mean", "max"])
        .round(6)
        .to_dict(orient="index")
    )
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "unique_user_ids": int(data["user_id"].nunique()),
        "unique_source_rows": int(data["source_row"].nunique()),
        "blank_counts": {name: int(count) for name, count in data.isna().sum().items()},
        "exact_duplicate_rows_after_first": int(data.duplicated().sum()),
        "zero_exposure_rows": int(data["total_ads"].eq(0).sum()),
        "observed_exposure_count_summary": exposure,
        "exposure_summary_role": "Descriptive only. Exposure count follows group assignment.",
    }


def write_group_table(results, path):
    """Export aggregate group conversion rates and interval bounds."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("group", "n", "converted", "observed_share", "rate", "wilson_95_lower", "wilson_95_upper"))
        for name in ("ad", "psa"):
            row = results["groups"][name]
            ci = row["wilson_95_ci"]
            writer.writerow(
                (name, row["n"], row["converted"], row["observed_share"], row["conversion_rate"], ci["lower"], ci["upper"])
            )


def run(raw_dir: Path, reports_dir: Path) -> None:
    """Run the primary group contrast and export only aggregates."""
    data, provenance = acquire_data(raw_dir)
    LOGGER.info("Validated %s exposed-user records", len(data))
    results = run_analysis(data)
    results["source"] = provenance
    reports_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".building-", dir=reports_dir) as temp:
        stage = Path(temp)
        for name, result in (("metrics.json", results), ("eda.json", explore(data))):
            (stage / name).write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        write_group_table(results, stage / "group-comparison.csv")
        write_figures(results, stage / "figures")
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                destination = reports_dir / file.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                file.replace(destination)

    LOGGER.info("Wrote conversion comparison and reports to %s", reports_dir)


def main():
    """Run a version-pinned analysis with optional cache and report locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
