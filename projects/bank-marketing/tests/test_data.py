"""Exercise the loader and pre-contact feature contract without UCI access."""

import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from src.data import (
    ARCHIVE_MEMBER,
    CACHE_NAME,
    CATEGORICAL_FEATURES,
    CSV_MEMBER,
    NUMERIC_FEATURES,
    acquire_data,
    features_and_target,
    validate_data,
)


def contact_rows() -> pd.DataFrame:
    first_contact = {
        "age": 35,
        "job": " UNKNOWN ",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": -85,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "mar",
        "duration": 134,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
        "y": " NO ",
    }
    follow_up = first_contact | {"pdays": 35, "previous": 2, "poutcome": "success", "y": "yes"}
    return pd.DataFrame([first_contact, follow_up], index=[17, 2])


def archive_containing(csv_data: bytes) -> bytes:
    nested = BytesIO()
    with ZipFile(nested, mode="w") as archive:
        archive.writestr(CSV_MEMBER, csv_data)
    outer = BytesIO()
    with ZipFile(outer, mode="w") as archive:
        archive.writestr(ARCHIVE_MEMBER, nested.getvalue())
    return outer.getvalue()


class ValidateDataTest(unittest.TestCase):
    def test_preserves_order_and_valid_negative_balance(self):
        source = contact_rows()
        data = validate_data(pd.concat([source, source.iloc[:1]]))
        self.assertEqual(data.index.tolist(), [17, 2, 17])
        self.assertEqual(data["balance"].tolist(), [-85, -85, -85])
        self.assertEqual(data["job"].tolist(), ["unknown", "unknown", "unknown"])

    def test_rejects_schema_changes_and_missing_values(self):
        frame = contact_rows()
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(frame.drop(columns="duration"))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(frame.assign(extra=1))
        frame.loc[17, "education"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_data(frame)

    def test_rejects_bad_targets_and_inconsistent_prior_contacts(self):
        with self.assertRaisesRegex(ValueError, "yes/no in y"):
            validate_data(contact_rows().assign(y="maybe"))
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_data(contact_rows().assign(previous=0))

    def test_allowlist_excludes_leakage_and_maps_target(self):
        source = validate_data(contact_rows())
        features, target = features_and_target(source)
        self.assertEqual(tuple(features), NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        self.assertEqual(features.index.tolist(), [17, 2])
        self.assertEqual(target.index.tolist(), features.index.tolist())
        self.assertEqual(target.tolist(), [0, 1])
        self.assertEqual(features["contacted_before"].tolist(), [0, 1])
        self.assertTrue(pd.isna(features.loc[17, "prior_days"]))
        self.assertEqual(features.loc[2, "prior_days"], 35)
        excluded = ["age", "marital", "contact", "day", "month", "duration", "campaign", "y"]
        self.assertTrue(features.columns.intersection(excluded).empty)


class AcquireDataTest(unittest.TestCase):
    def test_verifies_nested_archive_and_reuses_cache_offline(self):
        csv_data = contact_rows().to_csv(sep=";", index=False).encode()
        archive = archive_containing(csv_data)
        digest = sha256(csv_data).hexdigest()
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.CSV_SHA256", digest),
                patch("src.data.EXPECTED_ROWS", 2),
                patch("src.data.urlopen", return_value=BytesIO(archive)) as download,
            ):
                original, provenance = acquire_data(raw)
                cached, _ = acquire_data(raw)
            pd.testing.assert_frame_equal(original, cached)
            self.assertEqual(provenance["csv_sha256"], digest)
            self.assertEqual((raw / CACHE_NAME).read_bytes(), archive)
            download.assert_called_once()

    def test_checksum_failure_does_not_create_cache(self):
        archive = archive_containing(b"wrong release")
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.urlopen", return_value=BytesIO(archive)),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                acquire_data(raw)
            self.assertFalse((raw / CACHE_NAME).exists())
