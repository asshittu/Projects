"""Validate the UCI workbook loader and its raw-line contract with synthetic data."""

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
    RAW_COLUMNS,
    WORKBOOK_MEMBER,
    WORKBOOK_SHEET,
    acquire_data,
    validate_data,
)


def invoice_rows() -> pd.DataFrame:
    purchase = {
        "InvoiceNo": 700000,
        "StockCode": " A10 ",
        "Description": " Ceramic mug ",
        "Quantity": 2,
        "InvoiceDate": pd.Timestamp("2011-02-10T10:00:00"),
        "UnitPrice": 4.125,
        "CustomerID": 20001,
        "Country": " United Kingdom ",
    }
    cancellation = purchase | {
        "InvoiceNo": " C700000 ", "Quantity": -1, "Description": None, "CustomerID": None,
    }
    adjustment = purchase | {
        "InvoiceNo": "A700012", "Quantity": 0, "UnitPrice": -3.5, "Description": "NA",
    }
    return pd.DataFrame([purchase, cancellation, adjustment, purchase], index=[17, 3, 8, 12])


def workbook_archive(frame: pd.DataFrame) -> tuple[bytes, bytes]:
    workbook = BytesIO()
    frame.to_excel(workbook, sheet_name=WORKBOOK_SHEET, engine="openpyxl", index=False)
    workbook_data = workbook.getvalue()
    outer = BytesIO()
    with ZipFile(outer, mode="w") as archive:
        archive.writestr(WORKBOOK_MEMBER, workbook_data)
    return outer.getvalue(), workbook_data


class ValidateDataTest(unittest.TestCase):
    def test_normalizes_fields_without_filtering_or_reordering_lines(self):
        clean = validate_data(invoice_rows())
        self.assertEqual(tuple(clean), RAW_COLUMNS)
        self.assertEqual(clean.index.tolist(), [17, 3, 8, 12])
        self.assertEqual(clean.invoice_no.tolist(), ["700000", "C700000", "A700012", "700000"])
        self.assertEqual(clean.stock_code.tolist(), ["A10"] * 4)
        self.assertEqual(clean.country.tolist(), ["United Kingdom"] * 4)
        self.assertEqual(clean.quantity.tolist(), [2, -1, 0, 2])
        self.assertEqual(clean.unit_price.tolist(), [4.125, 4.125, -3.5, 4.125])
        self.assertEqual(clean.invoice_date.dtype, "datetime64[ns]")
        self.assertEqual(clean.duplicated().sum(), 1)

    def test_keeps_nullable_fields_and_literal_descriptions(self):
        clean = validate_data(invoice_rows())
        self.assertEqual(str(clean.customer_id.dtype), "Int64")
        self.assertEqual(clean.loc[17, "customer_id"], 20001)
        self.assertTrue(pd.isna(clean.loc[3, "customer_id"]))
        self.assertTrue(pd.isna(clean.loc[3, "description"]))
        self.assertEqual(clean.loc[8, "description"], "NA")

    def test_rejects_bad_schema_and_missing_required_fields(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(invoice_rows().drop(columns="InvoiceNo"))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_data(invoice_rows().assign(stock_code=""))
        with self.assertRaisesRegex(ValueError, "missing required values"):
            validate_data(invoice_rows().assign(Country="  "))

    def test_rejects_unparseable_values_in_typed_fields(self):
        for field, value in (
            ("Quantity", 1.5),
            ("UnitPrice", float("inf")),
            ("CustomerID", "invalid"),
            ("InvoiceDate", "invalid"),
            ("InvoiceDate", 44300),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_data(invoice_rows().assign(**{field: value}))


class AcquireDataTest(unittest.TestCase):
    def test_verifies_and_reads_workbook_and_reuses_cache(self):
        archive, workbook = workbook_archive(invoice_rows())
        digest = sha256(workbook).hexdigest()
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.WORKBOOK_SHA256", digest),
                patch("src.data.EXPECTED_ROWS", 4),
                patch("src.data.urlopen", return_value=BytesIO(archive)) as download,
            ):
                first, provenance = acquire_data(raw)
                cached, _ = acquire_data(raw)
            pd.testing.assert_frame_equal(first, cached)
            self.assertEqual(first.loc[2, "description"], "NA")
            self.assertTrue(pd.isna(first.loc[1, "customer_id"]))
            self.assertEqual(provenance["workbook_sha256"], digest)
            self.assertEqual((raw / CACHE_NAME).read_bytes(), archive)
            download.assert_called_once()

    def test_checksum_failure_leaves_no_cache(self):
        archive, _ = workbook_archive(invoice_rows())
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            with (
                patch("src.data.urlopen", return_value=BytesIO(archive)),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                acquire_data(raw)
            self.assertFalse((raw / CACHE_NAME).exists())
