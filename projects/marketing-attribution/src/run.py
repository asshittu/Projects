"""Generate aggregate attribution metrics, campaign credits, and a comparison chart."""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import duckdb

from .analysis import PROJECT, analyze
from .data import DATASET_URL, DOWNLOAD_URL, REVISION, SHA256, acquire_data, load_impressions


LOGGER = logging.getLogger(__name__)


def write_campaigns(path: Path, campaigns: list[dict]) -> None:
    """Write one record per campaign with descriptive credits and support counts."""
    fields = ("campaign", "touches", "paths", "first_touch", "last_touch", "linear_touch")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(campaigns)


def run(raw_dir: Path, reports_dir: Path, local_file: Path | None, lookback_days: int) -> None:
    """Validate source bytes, fit the reporting scope, and publish aggregate files."""
    source = acquire_data(raw_dir, local_file)
    LOGGER.info("Scanning the pinned Criteo release")
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as connection:
        load_impressions(connection, source)
        metrics, campaigns, same_user_campaigns = analyze(connection, lookback_days)
    if not campaigns:
        raise ValueError("The selected lookback yields no eligible conversion paths.")

    metrics["provenance"] = {
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "source_revision": REVISION,
        "gzip_sha256": SHA256,
        "analyzed_on": datetime.now(ZoneInfo("America/Toronto")).date().isoformat(),
        "source_license": "CC BY-NC-SA 4.0",
        "source_license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-", dir=reports_dir) as temp:
        stage = Path(temp)
        os.environ.setdefault("MPLCONFIGDIR", str(stage / "matplotlib-cache"))
        from .figures import write_figures

        (stage / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_campaigns(stage / "campaign-credit.csv", campaigns)
        write_campaigns(stage / "campaign-credit-same-user.csv", same_user_campaigns)
        write_figures(campaigns, metrics, stage)
        write_figures(same_user_campaigns, {
            "lookback_days": lookback_days,
            "paths": metrics["same_user_sensitivity"]["paths"],
            "scope_label": "Same-user sensitivity",
        }, stage, filename="campaign-credit-same-user.svg")
        write_report_terms(stage / "ATTRIBUTION.md")
        for name in (
            "metrics.json", "campaign-credit.csv", "campaign-credit.svg",
            "campaign-credit-same-user.csv", "campaign-credit-same-user.svg", "ATTRIBUTION.md",
        ):
            (stage / name).replace(reports_dir / name)
    LOGGER.info("Assigned credits for %s conversions", metrics["paths"]["qualified_conversions"])


def write_report_terms(path: Path) -> None:
    """Credit the source and state the scope of the report adaptations' license."""
    path.write_text(
        "# Criteo report attribution\n\n"
        "The report metrics, table, and chart adapt the Criteo Attribution Modeling for Bidding "
        "Dataset. The analysis selects qualifying impression paths and aggregates them by campaign. "
        "These report adaptations are shared under CC BY-NC-SA 4.0. The license applies to the "
        "reports derived from Criteo data.\n\n"
        f"[Criteo data and terms]({DATASET_URL}) · "
        "[Paper by Diemert, Meynet, Lefortier, and Galland](https://arxiv.org/abs/1707.06409) · "
        "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--source-file", type=Path, help="Read an already downloaded pinned gzip file")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "reports")
    parser.add_argument("--lookback-days", type=int, choices=range(1, 31), default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.reports_dir, args.source_file, args.lookback_days)


if __name__ == "__main__":
    main()
