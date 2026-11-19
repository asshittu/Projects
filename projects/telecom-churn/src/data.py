"""Acquire the original UCI Iranian Churn release and build retention features."""

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
CACHE_NAME = "iranian-churn.zip"
CSV_MEMBER = "Customer Churn.csv"
CSV_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
EXPECTED_ROWS = 3_150
MAX_ARCHIVE_BYTES = 512 * 1024
MAX_CSV_BYTES = 1024 * 1024

RAW_COLUMNS = (
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms", "distinct_called_numbers",
    "age_group", "tariff_plan", "status", "age", "customer_value", "churn",
)
ELIGIBLE_RAW_FEATURES = (
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms", "distinct_called_numbers",
    "tariff_plan",
)
CATEGORICAL_FEATURES = ("tariff_plan",)
NUMERIC_FEATURES = (
    *(name for name in ELIGIBLE_RAW_FEATURES if name not in CATEGORICAL_FEATURES),
    "average_call_seconds", "no_voice_calls",
)
CODE_VALUES = {
    "complains": (0, 1),
    "charge_amount": tuple(range(11)),
    "age_group": tuple(range(1, 6)),
    "tariff_plan": (1, 2),
    "status": (1, 2),
    "churn": (0, 1),
}


def _cache_archive(path: Path, archive: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(archive)
        pending.replace(path)


def acquire_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Download or reuse the official ZIP, verify `Customer Churn.csv`, and validate it."""
    cache_path = raw_dir / CACHE_NAME
    cached = cache_path.is_file()
    if cached:
        with cache_path.open("rb") as handle:
            archive = handle.read(MAX_ARCHIVE_BYTES + 1)
    else:
        with urlopen(DOWNLOAD_URL, timeout=30) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpected size for the UCI Iranian Churn archive.")

    with ZipFile(BytesIO(archive)) as source:
        member = source.getinfo(CSV_MEMBER)
        if member.file_size > MAX_CSV_BYTES:
            raise ValueError(f"Unexpected size for UCI archive member {CSV_MEMBER}.")
        csv_data = source.read(member)
    csv_digest = sha256(csv_data).hexdigest()
    if csv_digest != CSV_SHA256:
        raise ValueError(f"The UCI {CSV_MEMBER} SHA-256 does not match the pinned release.")

    data = validate_data(pd.read_csv(BytesIO(csv_data), encoding="utf-8"))
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} UCI records, found {len(data):,}.")
    if not cached:
        _cache_archive(cache_path, archive)

    provenance = {
        "dataset_id": 563,
        "dataset_name": "UCI Iranian Churn",
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "csv_member": CSV_MEMBER,
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "csv_sha256": csv_digest,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "rows": len(data),
    }
    return data, provenance


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source headers and coding while retaining every input row."""
    normalized = frame.rename(columns=lambda name: "_".join(str(name).lower().split()))
    missing = sorted(set(RAW_COLUMNS).difference(normalized.columns))
    extra = sorted(set(normalized.columns).difference(RAW_COLUMNS))
    if missing or extra or not normalized.columns.is_unique:
        raise ValueError(f"Unexpected Iranian Churn schema. Missing {missing}, extra {extra}.")
    if normalized.empty:
        raise ValueError("Iranian Churn data has no rows.")
    if normalized.isna().any().any():
        null_columns = normalized.columns[normalized.isna().any()].tolist()
        raise ValueError(f"Unexpected missing values in {null_columns}.")

    clean = normalized.loc[:, list(RAW_COLUMNS)].copy()
    for column in RAW_COLUMNS:
        values = pd.to_numeric(clean[column], errors="raise")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"Expected finite nonnegative values in {column}.")
        if column == "customer_value":
            clean[column] = values.astype("float64")
        else:
            if values.mod(1).ne(0).any():
                raise ValueError(f"Expected integer values in {column}.")
            clean[column] = values.astype("int64")

    for column, permitted in CODE_VALUES.items():
        if not clean[column].isin(permitted).all():
            raise ValueError(f"Unexpected code in {column}. Expected one of {permitted}.")
    if clean["seconds_of_use"].eq(0).ne(clean["frequency_of_use"].eq(0)).any():
        raise ValueError("Voice usage seconds and call count disagree about zero usage.")

    return clean


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the retention feature allowlist to a validated raw frame."""
    features = frame.loc[:, list(ELIGIBLE_RAW_FEATURES)].copy()
    call_count = features["frequency_of_use"]
    no_calls = call_count.eq(0)
    features["average_call_seconds"] = features["seconds_of_use"].div(call_count.mask(no_calls))
    features["no_voice_calls"] = no_calls.astype("int8")
    target = frame["churn"].astype("int8")
    return features.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]], target


def profile_groups(frame: pd.DataFrame) -> pd.Series:
    """Group identical observed feature profiles while keeping all original rows."""
    return frame.groupby(list(ELIGIBLE_RAW_FEATURES), sort=False).ngroup().rename("profile_group")
