"""Verify that the notebook handoff contains only aggregate evidence."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.report import write_report


class ReportTests(unittest.TestCase):
    def test_archive_is_allowlisted_and_repeatable(self):
        fixture = {
            "folds": [{
                "name": "holdout",
                "n_evaluated": 8,
                "models": {
                    "seasonal_naive": {"mae": 2, "rmsle": 0.2},
                    "hist_gradient_boosting": {"mae": 1, "rmsle": 0.1},
                },
            }]
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "air_visit_data.csv").write_text("fixture", encoding="utf-8")
            output = write_report(fixture, root)
            before = output.read_bytes()
            with ZipFile(output) as archive:
                self.assertEqual(
                    set(archive.namelist()), {"metrics.json", "mae-by-window.svg", "rmsle-by-window.svg"},
                )
                self.assertEqual(json.loads(archive.read("metrics.json")), fixture)
                self.assertIn("<svg", archive.read("mae-by-window.svg").decode())
            write_report(fixture, root)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
