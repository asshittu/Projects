"""Validate the UCI workbook and export aggregate DuckDB retail marts."""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import acquire_data
from .figures import write_figures
from .marts import build_marts

PROJECT = Path(__file__).resolve().parents[1]
CUSTOMER_CUTOFF = date(2011, 12, 1)
LOGGER = logging.getLogger(__name__)


def run(raw_dir: Path, reports_dir: Path) -> None:
    """Build and export a reconciled ledger, RFM, and repeat-activity analysis."""
    lines, provenance = acquire_data(raw_dir)
    LOGGER.info("Validated %s source invoice lines", len(lines))
    marts = build_marts(lines, cutoff=CUSTOMER_CUTOFF)
    summary = {
        "source": provenance,
        "customer_cutoff": CUSTOMER_CUTOFF.isoformat(),
        "exact_repeated_source_rows_after_first": int(lines.duplicated().sum()),
        "summary": json.loads(marts["summary"].to_json(orient="records", date_format="iso"))[0],
        "quality": json.loads(marts["quality"].to_json(orient="records", date_format="iso")),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".building-", dir=reports_dir) as temp:
        stage = Path(temp)
        (stage / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name in ("monthly", "segments", "cohorts", "quality"):
            marts[name].to_csv(
                stage / f"{name}.csv", index=False, lineterminator="\n", date_format="%Y-%m-%d",
            )
        write_figures(marts, stage / "figures")
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                destination = reports_dir / file.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                file.replace(destination)

    LOGGER.info("Wrote monthly, segmentation and cohort reports to %s", reports_dir)


def main():
    """Run the analysis with optional cache and report directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
