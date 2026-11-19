"""Test official archive handling and assignment checks with synthetic rows."""

import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from src.data import CACHE_NAME, CSV_MEMBER, RAW_COLUMNS, acquire_data, validate_data


def campaign_rows() -> pd.DataFrame:
    exposed = {
        "Unnamed: 0": 17,
        "user id": 987654,
        "test group": " AD ",
        "converted": " TRUE ",
        "total ads": 15,
        "most ads day": " monday ",
        "most ads hour": 23,
    }
    control = exposed | {
        "Unnamed: 0": 3, "user id": 987655, "test group": "psa", "converted": False,
    }
    another_exposed = exposed | {"Unnamed: 0": 8, "user id": 987656, "converted": False}
    return pd.DataFrame([exposed, control, another_exposed], index=[17, 3, 8])


def archive_containing(csv_data: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(CSV_MEMBER, csv_data)
    return buffer.getvalue()


class ValidateDataTest(unittest.TestCase):
    def test_normalizes_names_and_codes_without_reordering_rows(self):
        clean = validate_data(campaign_rows())
        self.assertEqual(tuple(clean), RAW_COLUMNS)
        self.assertEqual(clean.index.tolist(), [17, 3, 8])
        self.assertEqual(clean["test_group"].tolist(), ["ad", "psa", "ad"])
        self.assertEqual(clean["converted"].tolist(), [True, False, False])
        self.assertTrue(pd.api.types.is_bool_dtype(clean["converted"]))
        self.assertEqual(clean["most_ads_day"].tolist(), ["Monday"] * 3)

    def test_rejects_bad_schema_and_missing_values(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(campaign_rows().drop(columns="test group"))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(campaign_rows().assign(test_group="ad"))
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_data(campaign_rows().assign(converted=None))

    def test_rejects_invalid_outcome_group_and_exposure_coding(self):
        for field, value in (
            ("converted", "maybe"),
            ("test group", "organic"),
            ("total ads", -2),
            ("most ads day", "holiday"),
            ("most ads hour", 24),
            ("most ads hour", 2.5),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_data(campaign_rows().assign(**{field: value}))

    def test_reports_cross_group_reuse_with_aggregate_counts(self):
        frame = campaign_rows()
        frame.loc[3, "user id"] = frame.loc[17, "user id"]
        with self.assertRaisesRegex(ValueError, "1 IDs assigned to both groups") as error:
            validate_data(frame)
        self.assertNotIn("987654", str(error.exception))

    def test_rejects_duplicate_user_rows_in_same_group(self):
        frame = campaign_rows()
        frame.loc[8, "user id"] = frame.loc[17, "user id"]
        with self.assertRaisesRegex(ValueError, "2 rows with repeated user IDs"):
            validate_data(frame)


class AcquireDataTest(unittest.TestCase):
    def test_verifies_source_member_and_reuses_cache(self):
        csv_data = campaign_rows().to_csv(index=False).encode()
        archive = archive_containing(csv_data)
        digest = sha256(csv_data).hexdigest()
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.CSV_SHA256", digest),
                patch("src.data.EXPECTED_ROWS", 3),
                patch("src.data.urlopen", return_value=BytesIO(archive)) as download,
            ):
                original, provenance = acquire_data(raw)
                cached, _ = acquire_data(raw)
            pd.testing.assert_frame_equal(original, cached)
            self.assertEqual(provenance["csv_sha256"], digest)
            self.assertEqual((raw / CACHE_NAME).read_bytes(), archive)
            download.assert_called_once()

    def test_corrupt_and_wrong_release_downloads_leave_no_cache(self):
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            for content, expected_error in (
                (b"<html>unexpected page</html>", "Expected the Kaggle ZIP"),
                (archive_containing(b"wrong release"), "SHA-256"),
            ):
                with (
                    patch("src.data.urlopen", return_value=BytesIO(content)),
                    self.assertRaisesRegex(ValueError, expected_error),
                ):
                    acquire_data(raw)
                self.assertFalse((raw / CACHE_NAME).exists())
