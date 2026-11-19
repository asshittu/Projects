"""Check the UCI loader and leakage-safe feature contract offline."""

import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from src.data import (
    CACHE_NAME,
    CATEGORICAL_FEATURES,
    CSV_MEMBER,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    acquire_data,
    features_and_target,
    profile_groups,
    validate_data,
)


def customer_rows() -> pd.DataFrame:
    no_voice = {
        "Call  Failure": 0,
        "Complains": 0,
        "Subscription  Length": 12,
        "Charge  Amount": 10,
        "Seconds of Use": 0,
        "Frequency of use": 0,
        "Frequency of SMS": 15,
        "Distinct Called Numbers": 0,
        "Age Group": 1,
        "Tariff Plan": 2,
        "Status": 2,
        "Age": 15,
        "Customer Value": 0.25,
        "Churn": 0,
    }
    same_profile = no_voice | {
        "Age Group": 2, "Status": 1, "Age": 25, "Customer Value": 1.25, "Churn": 1,
    }
    voice_user = no_voice | {"Seconds of Use": 90, "Frequency of use": 4, "Churn": 1}
    return pd.DataFrame([no_voice, same_profile, voice_user], index=[17, 2, 11])


def archive_containing(csv_data: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(CSV_MEMBER, csv_data)
    return buffer.getvalue()


class ValidateDataTest(unittest.TestCase):
    def test_normalizes_headers_and_preserves_order_and_rows(self):
        source = customer_rows()
        clean = validate_data(pd.concat([source, source.iloc[:1]]))
        self.assertEqual(tuple(clean), RAW_COLUMNS)
        self.assertEqual(clean.index.tolist(), [17, 2, 11, 17])
        self.assertEqual(clean["charge_amount"].tolist(), [10, 10, 10, 10])
        self.assertEqual(clean["customer_value"].tolist(), [0.25, 1.25, 0.25, 0.25])

    def test_rejects_bad_schema_and_missing_values(self):
        frame = customer_rows()
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(frame.drop(columns="Status"))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(frame.assign(status=1))
        frame.loc[17, "Customer Value"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_data(frame)

    def test_rejects_invalid_codes_and_inconsistent_voice_fields(self):
        with self.assertRaisesRegex(ValueError, "Unexpected code in churn"):
            validate_data(customer_rows().assign(Churn=2))
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_data(customer_rows().assign(**{"Seconds of Use": 99}))

    def test_builds_features_and_aligned_binary_target(self):
        source = validate_data(customer_rows())
        features, target = features_and_target(source)
        self.assertEqual(tuple(features), NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        self.assertEqual(target.index.tolist(), features.index.tolist())
        self.assertEqual(target.tolist(), [0, 1, 1])
        self.assertEqual(features["no_voice_calls"].tolist(), [1, 1, 0])
        self.assertTrue(pd.isna(features.loc[17, "average_call_seconds"]))
        self.assertEqual(features.loc[11, "average_call_seconds"], 22.5)
        excluded = ["status", "customer_value", "age", "age_group", "churn"]
        self.assertTrue(features.columns.intersection(excluded).empty)

    def test_profile_group_ignores_excluded_fields_and_label(self):
        source = validate_data(customer_rows())
        groups = profile_groups(source)
        self.assertEqual(groups.index.tolist(), [17, 2, 11])
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])


class AcquireDataTest(unittest.TestCase):
    def test_verifies_source_member_and_reuses_cache(self):
        csv_data = customer_rows().to_csv(index=False).encode()
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

    def test_checksum_failure_leaves_no_cache(self):
        archive = archive_containing(b"wrong release")
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.urlopen", return_value=BytesIO(archive)),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                acquire_data(raw)
            self.assertFalse((raw / CACHE_NAME).exists())
