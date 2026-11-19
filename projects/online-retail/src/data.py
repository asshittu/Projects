"""Acquire and normalize the original UCI Online Retail workbook."""

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
CACHE_NAME = "online-retail.zip"
WORKBOOK_MEMBER = "Online Retail.xlsx"
WORKBOOK_SHEET = "Online Retail"
WORKBOOK_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
EXPECTED_ROWS = 541_909
MAX_ARCHIVE_BYTES = 32 * 1024**2
MAX_WORKBOOK_BYTES = 32 * 1024**2

SOURCE_COLUMNS = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}
RAW_COLUMNS = tuple(SOURCE_COLUMNS.values())
REQUIRED_COLUMNS = ("invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "country")


def _cache_archive(path: Path, archive: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(archive)
        pending.replace(path)


def acquire_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Fetch or reuse UCI's archive, verify the workbook, and normalize all rows."""
    cache_path = raw_dir / CACHE_NAME
    cached = cache_path.is_file()
    if cached:
        with cache_path.open("rb") as handle:
            archive = handle.read(MAX_ARCHIVE_BYTES + 1)
    else:
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpected size for the UCI Online Retail archive.")

    try:
        with ZipFile(BytesIO(archive)) as source:
            member = source.getinfo(WORKBOOK_MEMBER)
            if member.file_size > MAX_WORKBOOK_BYTES:
                raise ValueError(f"Unexpected size for UCI archive member {WORKBOOK_MEMBER}.")
            workbook = source.read(member)
    except (BadZipFile, KeyError) as error:
        raise ValueError(f"Expected the UCI ZIP archive with member {WORKBOOK_MEMBER}.") from error

    workbook_digest = sha256(workbook).hexdigest()
    if workbook_digest != WORKBOOK_SHA256:
        raise ValueError(f"The UCI {WORKBOOK_MEMBER} SHA-256 does not match the pinned release.")
    frame = pd.read_excel(
        BytesIO(workbook), sheet_name=WORKBOOK_SHEET, engine="openpyxl",
        keep_default_na=False, na_values=[""],
    )
    data = validate_data(frame)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} UCI records, found {len(data):,}.")
    if not cached:
        _cache_archive(cache_path, archive)

    provenance = {
        "dataset_id": 352,
        "dataset_name": "UCI Online Retail",
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "workbook_member": WORKBOOK_MEMBER,
        "workbook_sheet": WORKBOOK_SHEET,
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "workbook_sha256": workbook_digest,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "rows": len(data),
    }
    return data, provenance


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the source schema while preserving rows and quality signals for SQL."""
    normalized = frame.rename(columns=SOURCE_COLUMNS)
    missing = sorted(set(RAW_COLUMNS).difference(normalized.columns))
    extra = sorted(set(normalized.columns).difference(RAW_COLUMNS), key=str)
    if missing or extra or not normalized.columns.is_unique:
        raise ValueError(f"Unexpected Online Retail schema. Missing {missing}, extra {extra}.")
    if normalized.empty:
        raise ValueError("Online Retail data has no rows.")

    data = normalized.loc[:, list(RAW_COLUMNS)].copy()
    for column in ("invoice_no", "stock_code", "description", "country"):
        values = data[column].astype("string").str.strip()
        data[column] = values.mask(values.eq(""))
    missing_required = data.loc[:, list(REQUIRED_COLUMNS)].isna().any()
    if missing_required.any():
        raise ValueError(f"Unexpected missing required values in {missing_required[missing_required].index.tolist()}.")

    quantity = pd.to_numeric(data["quantity"], errors="coerce")
    if not np.isfinite(quantity).all() or quantity.mod(1).ne(0).any():
        raise ValueError("Expected finite integers in quantity.")
    data["quantity"] = quantity.astype("int64")

    price = pd.to_numeric(data["unit_price"], errors="coerce")
    if not np.isfinite(price).all():
        raise ValueError("Expected finite values in unit_price.")
    data["unit_price"] = price.astype("float64")

    values = data["customer_id"].astype("string").str.strip()
    values = values.mask(values.eq(""))
    customer = pd.to_numeric(values, errors="coerce")
    identified = customer.loc[values.notna()]
    if identified.isna().any() or not np.isfinite(identified).all() or identified.mod(1).ne(0).any():
        raise ValueError("Expected finite integer IDs or missing values in customer_id.")
    data["customer_id"] = customer.astype("Int64")

    raw_dates = data["invoice_date"]
    if pd.api.types.is_numeric_dtype(raw_dates):
        raise ValueError("Expected invoice timestamps in invoice_date.")
    dates = pd.to_datetime(raw_dates, format="ISO8601", errors="coerce")
    if dates.isna().any() or not pd.api.types.is_datetime64_dtype(dates):
        raise ValueError("Expected valid timezone-naive timestamps in invoice_date.")
    data["invoice_date"] = dates.astype("datetime64[ns]")

    return data
