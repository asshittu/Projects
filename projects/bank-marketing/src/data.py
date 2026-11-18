"""Acquire the original UCI Bank Marketing data and build pre-call features."""

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CACHE_NAME = "bank-marketing.zip"
ARCHIVE_MEMBER = "bank.zip"
CSV_MEMBER = "bank-full.csv"
CSV_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
EXPECTED_ROWS = 45_211
MAX_ARCHIVE_BYTES = 4 * 1024**2
MAX_CSV_BYTES = 8 * 1024**2

RAW_COLUMNS = (
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
)
RAW_NUMERIC = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")
RAW_CATEGORICAL = tuple(column for column in RAW_COLUMNS if column not in RAW_NUMERIC)
NUMERIC_FEATURES = ("balance", "previous", "contacted_before", "prior_days")
CATEGORICAL_FEATURES = ("job", "education", "default", "housing", "loan", "poutcome")
TARGET_VALUES = {"no": 0, "yes": 1}


def _read_member(archive: bytes, name: str, limit: int) -> bytes:
    with ZipFile(BytesIO(archive)) as source:
        member = source.getinfo(name)
        if member.file_size > limit:
            raise ValueError(f"Unexpected size for UCI archive member {name}.")
        return source.read(member)


def _cache_archive(path: Path, archive: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(archive)
        pending.replace(path)


def acquire_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Fetch or reuse the official archive, verify its CSV, and retain source row order."""
    cache_path = raw_dir / CACHE_NAME
    cached = cache_path.is_file()
    if cached:
        archive = cache_path.read_bytes()
    else:
        with urlopen(DOWNLOAD_URL, timeout=30) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpected size for the UCI Bank Marketing archive.")

    nested_archive = _read_member(archive, ARCHIVE_MEMBER, MAX_ARCHIVE_BYTES)
    csv_data = _read_member(nested_archive, CSV_MEMBER, MAX_CSV_BYTES)
    csv_digest = sha256(csv_data).hexdigest()
    if csv_digest != CSV_SHA256:
        raise ValueError(f"The UCI {CSV_MEMBER} SHA-256 does not match the pinned release.")

    data = validate_data(pd.read_csv(BytesIO(csv_data), sep=";", encoding="utf-8"))
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} UCI records, found {len(data):,}.")
    if not cached:
        _cache_archive(cache_path, archive)

    provenance = {
        "dataset_id": 222,
        "dataset_name": "UCI Bank Marketing",
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "archive_member": ARCHIVE_MEMBER,
        "csv_member": CSV_MEMBER,
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "csv_sha256": csv_digest,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "rows": len(data),
    }
    return data, provenance


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize types, validate source coding, and preserve each recorded contact."""
    missing = sorted(set(RAW_COLUMNS).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(RAW_COLUMNS))
    if missing or extra or not frame.columns.is_unique:
        raise ValueError(f"Unexpected Bank Marketing schema. Missing {missing}, extra {extra}.")
    if frame.empty:
        raise ValueError("Bank Marketing data has no rows.")
    if frame.isna().any().any():
        null_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Unexpected missing values in {null_columns}.")

    clean = frame.loc[:, list(RAW_COLUMNS)].copy()
    for column in RAW_NUMERIC:
        values = pd.to_numeric(clean[column], errors="raise")
        if not np.isfinite(values).all() or values.mod(1).ne(0).any():
            raise ValueError(f"Expected finite integers in {column}.")
        clean[column] = values.astype("int64")

    for column in RAW_CATEGORICAL:
        values = clean[column].astype("string").str.strip().str.lower()
        if values.eq("").any():
            raise ValueError(f"Unexpected empty category in {column}.")
        clean[column] = values

    for column in ("default", "housing", "loan", "y"):
        if not clean[column].isin(TARGET_VALUES).all():
            raise ValueError(f"Expected yes/no in {column}.")
    minimums = {"age": 18, "duration": 0, "campaign": 1, "pdays": -1, "previous": 0}
    for column, minimum in minimums.items():
        if clean[column].lt(minimum).any():
            raise ValueError(f"Values in {column} must be at least {minimum}.")
    if not clean["day"].between(1, 31).all():
        raise ValueError("Contact day must be between 1 and 31.")
    if clean["pdays"].eq(-1).ne(clean["previous"].eq(0)).any():
        raise ValueError("pdays and previous disagree about prior contact history.")

    return clean


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select an explicit allowlist from validated rows and map the subscription label."""
    features = frame.loc[:, ["balance", "previous", *CATEGORICAL_FEATURES]].copy()
    features["contacted_before"] = frame["pdays"].ne(-1).astype("int8")
    features["prior_days"] = frame["pdays"].where(features["contacted_before"].eq(1))
    target = frame["y"].map(TARGET_VALUES)
    if target.isna().any():
        raise ValueError("Expected yes/no in y.")
    return features.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]], target.astype("int8")
