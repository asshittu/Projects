"""Exercise the private-input contract without any competition files."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data import find_competition_directory, load_competition_data

SYNTHETIC_INPUTS = {
    "air_visit_data.csv": (
        "air_store_id,visit_date,visitors\n"
        "air_demo_a,2025-01-01,10\n"
        "air_demo_a,2025-01-02,0\n"
        "air_demo_b,2025-01-02,8\n"
    ),
    "date_info.csv": (
        "calendar_date,day_of_week,holiday_flg\n"
        "2025-01-01,Wednesday,1\n"
        "2025-01-02,Thursday,0\n"
        "2025-01-03,Friday,0\n"
        "2025-01-04,Saturday,0\n"
    ),
    "air_reserve.csv": (
        "air_store_id,visit_datetime,reserve_datetime,reserve_visitors\n"
        "air_demo_a,2025-01-04 18:00:00,2025-01-02 12:00:00,2\n"
        "air_demo_a,2025-01-04 18:00:00,2025-01-02 12:00:00,2\n"
    ),
    "sample_submission.csv": (
        "id,visitors\n"
        "air_demo_a_2025-01-03,0\n"
        "air_demo_a_2025-01-04,0\n"
        "air_demo_b_2025-01-03,0\n"
        "air_demo_b_2025-01-04,0\n"
    ),
}


def write_synthetic_csvs(directory: Path, overrides: dict[str, str] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, csv in (SYNTHETIC_INPUTS | (overrides or {})).items():
        (directory / filename).write_text(csv, encoding="utf-8")


class LoadCompetitionDataTest(unittest.TestCase):
    def test_preserves_records_and_derives_only_horizon_length(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_csvs(root)
            visits, calendar, reservations, horizon_length = load_competition_data(root)
        self.assertEqual(horizon_length, 2)
        self.assertEqual(tuple(visits), ("air_store_id", "visit_date", "visitors"))
        self.assertEqual(tuple(calendar), ("visit_date", "holiday_flg"))
        self.assertEqual(tuple(reservations), (
            "air_store_id", "visit_datetime", "reserve_datetime", "reserve_visitors", "visit_date",
        ))
        self.assertEqual(visits["visitors"].tolist(), [10, 0, 8])
        self.assertEqual(visits["visitors"].dtype, "int64")
        self.assertEqual(reservations["visit_date"].tolist(), [pd.Timestamp("2025-01-04")] * 2)
        self.assertEqual(reservations["reserve_visitors"].tolist(), [2, 2])

    def test_accepts_empty_reservations_with_expected_columns(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            header = SYNTHETIC_INPUTS["air_reserve.csv"].splitlines()[0] + "\n"
            write_synthetic_csvs(root, {"air_reserve.csv": header})
            _, _, reservations, _ = load_competition_data(root)
        self.assertTrue(reservations.empty)
        self.assertEqual(reservations["reserve_datetime"].dtype, "datetime64[ns]")

    def test_rejects_duplicate_history_or_missing_required_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = SYNTHETIC_INPUTS["air_visit_data.csv"]
            write_synthetic_csvs(root, {"air_visit_data.csv": original + "air_demo_a,2025-01-01,12\n"})
            with self.assertRaisesRegex(ValueError, "one recorded visit count"):
                load_competition_data(root)
            write_synthetic_csvs(root, {"air_visit_data.csv": original.replace("visitors", "unknown")})
            with self.assertRaisesRegex(ValueError, "readable air_visit_data.csv with columns"):
                load_competition_data(root)

    def test_rejects_invalid_forecast_horizon(self):
        original = SYNTHETIC_INPUTS["sample_submission.csv"]
        for value, message in (
            (original.replace("2025-01-04", "2025-01-05"), "continuous daily horizon"),
            (original.replace("2025-01-03", "2025-01-01"), "continuous daily horizon"),
            (original.replace("2025-01-03", "2025-01-02").replace("2025-01-04", "2025-01-03"), "follow all recorded"),
            (original + "air_demo_a_2025-01-03,0\n", "unique IDs"),
            (original.replace("_2025-01-03", "-2025-01-03"), "end with _YYYY-MM-DD"),
        ):
            with self.subTest(message=message), TemporaryDirectory() as directory:
                root = Path(directory)
                write_synthetic_csvs(root, {"sample_submission.csv": value})
                with self.assertRaisesRegex(ValueError, message):
                    load_competition_data(root)

    def test_rejects_missing_calendar_coverage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calendar = SYNTHETIC_INPUTS["date_info.csv"].replace("2025-01-04,Saturday,0\n", "")
            write_synthetic_csvs(root, {"date_info.csv": calendar})
            with self.assertRaisesRegex(ValueError, "calendar coverage"):
                load_competition_data(root)

    def test_drops_late_booking_rows_with_aggregate_warning(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = SYNTHETIC_INPUTS["air_reserve.csv"] + "air_demo_b,2025-01-04 18:00:00,2025-01-05 12:00:00,3\n"
            write_synthetic_csvs(root, {"air_reserve.csv": inputs})
            with self.assertLogs("src.data", level="WARNING") as logs:
                _, _, reservations, _ = load_competition_data(root)
        self.assertEqual(reservations["reserve_visitors"].tolist(), [2, 2])
        self.assertIn("Ignored post-visit reservation records (count 1).", logs.output[0])

    def test_rejects_invalid_dates_and_noninteger_or_negative_counts(self):
        original = SYNTHETIC_INPUTS["air_visit_data.csv"]
        for after, message in (
            ("2025-02-30,10", "valid timezone-naive dates"),
            ("2025-01-01,-10", "nonnegative integer counts"),
            ("2025-01-01,10.5", "nonnegative integer counts"),
            ("2025-01-01,inf", "nonnegative integer counts"),
        ):
            with self.subTest(after=after), TemporaryDirectory() as directory:
                root = Path(directory)
                write_synthetic_csvs(root, {"air_visit_data.csv": original.replace("2025-01-01,10", after)})
                with self.assertRaisesRegex(ValueError, message):
                    load_competition_data(root)


class FindCompetitionDirectoryTest(unittest.TestCase):
    def test_finds_one_complete_folder_beneath_input_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "competition" / "extracted"
            write_synthetic_csvs(expected)
            self.assertEqual(find_competition_directory(root), expected)
            self.assertEqual(find_competition_directory(expected), expected)

    def test_rejects_missing_or_ambiguous_matches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No complete"):
                find_competition_directory(root)
            write_synthetic_csvs(root / "first")
            write_synthetic_csvs(root / "second")
            with self.assertRaisesRegex(ValueError, "Multiple complete"):
                find_competition_directory(root)
