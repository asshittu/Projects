"""Test Yelp source validation and normalized deduplication with synthetic text."""

import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from src.data import CACHE_NAME, SOURCE_MEMBER, acquire_data, text_keys, validate_data


def review_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "text": ["  Ｇood   STRAßE  ", "good strasse", "Unhelpful synthetic reply."],
        "label": [1, 1, 0],
    }, index=[19, 7, 4])


def archive_containing(source: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(SOURCE_MEMBER, source)
    return buffer.getvalue()


class ValidateDataTest(unittest.TestCase):
    def test_normalized_comparison_keeps_the_first_original_text(self):
        raw = review_rows()
        self.assertEqual(text_keys(raw["text"]).tolist()[:2], ["good strasse"] * 2)
        data = validate_data(raw)
        self.assertEqual(data["text"].tolist(), [raw.iloc[0].text, raw.iloc[2].text])
        self.assertEqual(data["label"].tolist(), [1, 0])
        self.assertEqual(data["label"].dtype, "int8")
        self.assertEqual(data.index.tolist(), [0, 1])

    def test_rejects_same_normalized_text_with_conflicting_labels(self):
        raw = review_rows()
        raw.loc[7, "label"] = 0
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            validate_data(raw)

    def test_rejects_bad_schema_invalid_or_missing_values_and_blank_text(self):
        with self.assertRaisesRegex(ValueError, "nonempty review data"):
            validate_data(review_rows().assign(other=1))
        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_data(review_rows().assign(text=None))
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_data(review_rows().assign(label=3))
        with self.assertRaisesRegex(ValueError, "nonblank"):
            validate_data(review_rows().assign(text="   "))
        with self.assertRaisesRegex(ValueError, "strings"):
            validate_data(review_rows().assign(text=123))


class AcquireDataTest(unittest.TestCase):
    def test_checks_member_digest_and_reuses_cache_with_provenance(self):
        source = " Ｇreat service.  \t1\nGREAT SERVICE.\t1\nBland sample.\t0\n".encode()
        archive = archive_containing(source)
        digest = sha256(source).hexdigest()
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.SOURCE_SHA256", digest),
                patch("src.data.EXPECTED_ROWS", 3),
                patch("src.data.urlopen", return_value=BytesIO(archive)) as download,
            ):
                first, provenance = acquire_data(raw)
                cached, _ = acquire_data(raw)
            pd.testing.assert_frame_equal(first, cached)
            self.assertEqual(first.loc[0, "text"], " Ｇreat service.  ")
            self.assertEqual(provenance["source_sha256"], digest)
            self.assertEqual(provenance["raw_label_counts"], {"0": 1, "1": 2})
            self.assertEqual(provenance["retained_label_counts"], {"0": 1, "1": 1})
            self.assertEqual(provenance["duplicates_removed"], 1)
            self.assertEqual((raw / CACHE_NAME).read_bytes(), archive)
            download.assert_called_once()

    def test_checksum_mismatch_leaves_no_cache(self):
        archive = archive_containing(b"Unexpected unpublished fixture.\t1\n")
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.urlopen", return_value=BytesIO(archive)),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                acquire_data(raw)
            self.assertFalse((raw / CACHE_NAME).exists())
