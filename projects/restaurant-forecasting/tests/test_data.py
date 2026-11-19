"""Exercise the private-input contract without any competition files."""

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

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


def synthetic_zip(entries: dict[str, bytes | str]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class CompressedInputsTest(unittest.TestCase):
    def assert_loads_fixture(self, input_path: Path) -> None:
        visits, calendar, reservations, horizon = load_competition_data(input_path)
        self.assertEqual(visits["visitors"].tolist(), [10, 0, 8])
        self.assertEqual(calendar["holiday_flg"].tolist(), [1, 0, 0, 0])
        self.assertEqual(reservations["reserve_visitors"].tolist(), [2, 2])
        self.assertEqual(horizon, 2)

    def test_loads_individual_csv_zip_files_under_input_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "competition" / "download"
            data.mkdir(parents=True)
            for name, csv in SYNTHETIC_INPUTS.items():
                (data / f"{name}.zip").write_bytes(synthetic_zip({f"nested/{name}": csv}))
            self.assert_loads_fixture(root)
            self.assertEqual(len(list(data.iterdir())), 4)

    def test_loads_an_outer_zip_with_csvs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "competition.zip"
            path.write_bytes(synthetic_zip({f"nested/{name}": csv for name, csv in SYNTHETIC_INPUTS.items()}))
            self.assert_loads_fixture(path)
            self.assert_loads_fixture(root)
            self.assertEqual(list(root.iterdir()), [path])

    def test_loads_outer_zip_with_nested_csv_zip_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "competition.zip"
            entries = {
                f"packed/{name}.zip": synthetic_zip({f"nested/{name}": csv})
                for name, csv in SYNTHETIC_INPUTS.items()
            }
            path.write_bytes(synthetic_zip(entries))
            self.assert_loads_fixture(path)
            self.assertEqual(list(root.iterdir()), [path])

    def test_loads_unique_mix_of_loose_csvs_and_zip_members(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = list(SYNTHETIC_INPUTS)
            (root / names[0]).write_text(SYNTHETIC_INPUTS[names[0]], encoding="utf-8")
            (root / f"{names[1]}.zip").write_bytes(synthetic_zip({names[1]: SYNTHETIC_INPUTS[names[1]]}))
            (root / "mixed.zip").write_bytes(synthetic_zip({
                f"mix/{names[2]}": SYNTHETIC_INPUTS[names[2]],
                f"mix/{names[3]}.zip": synthetic_zip({names[3]: SYNTHETIC_INPUTS[names[3]]}),
            }))
            self.assert_loads_fixture(root)

    def test_reports_missing_files_and_rejects_similar_basenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "competition.zip"
            with self.assertRaisesRegex(FileNotFoundError, "Missing required files") as empty:
                load_competition_data(root)
            path.write_bytes(synthetic_zip({
                f"wrong_{name}": csv for name, csv in SYNTHETIC_INPUTS.items()
            }))
            with self.assertRaisesRegex(FileNotFoundError, "Missing required files") as raised:
                load_competition_data(path)
            for name in SYNTHETIC_INPUTS:
                self.assertIn(name, str(empty.exception))
                self.assertIn(name, str(raised.exception))

    def test_rejects_multiple_exact_basenames_in_zip_members(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "competition.zip"
            name = "air_visit_data.csv"
            path.write_bytes(synthetic_zip(SYNTHETIC_INPUTS | {f"duplicated/{name}": SYNTHETIC_INPUTS[name]}))
            with self.assertRaisesRegex(ValueError, r"Multiple copies found for air_visit_data\.csv"):
                load_competition_data(path)

    def test_rejects_duplicate_inputs_across_loose_and_zip_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_csvs(root)
            name = "air_visit_data.csv"
            (root / f"{name}.zip").write_bytes(synthetic_zip({name: SYNTHETIC_INPUTS[name]}))
            with self.assertRaisesRegex(ValueError, r"Multiple copies found for air_visit_data\.csv"):
                load_competition_data(root)

    def test_rejects_oversized_zipped_csv_and_broken_inner_zip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "competition.zip"
            path.write_bytes(synthetic_zip(SYNTHETIC_INPUTS))
            with (
                patch("src.data.MAX_CSV_BYTES", 20),
                self.assertRaisesRegex(ValueError, "exceeds the supported read size"),
            ):
                load_competition_data(path)
            path.write_bytes(synthetic_zip({"air_visit_data.csv.zip": b"not a ZIP"}))
            with self.assertRaisesRegex(ValueError, r"readable nested ZIP input for air_visit_data\.csv"):
                load_competition_data(path)


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
