"""Download and validate Kaggle's original Marketing A/B Testing dataset."""

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

DATASET_ID = "faviovaz/marketing-ab-testing"
DATASET_VERSION = 1
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_ID}"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_ID}"
    f"?datasetVersionNumber={DATASET_VERSION}"
)
CACHE_NAME = "marketing-ab-testing.zip"
CSV_MEMBER = "marketing_AB.csv"
CSV_SHA256 = "de6ce8def1a6559e9e5ab364fce07cfe2e84f02dd407c3586eea55743c7e17f2"
EXPECTED_ROWS = 588_101
MAX_ARCHIVE_BYTES = 8 * 1024**2
MAX_CSV_BYTES = 32 * 1024**2

SOURCE_COLUMNS = {
    "Unnamed: 0": "source_row",
    "user id": "user_id",
    "test group": "test_group",
    "converted": "converted",
    "total ads": "total_ads",
    "most ads day": "most_ads_day",
    "most ads hour": "most_ads_hour",
}
RAW_COLUMNS = tuple(SOURCE_COLUMNS.values())
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
GROUPS = ("ad", "psa")
BOOLEAN_VALUES = {"true": True, "false": False}


def _cache_archive(path: Path, archive: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(archive)
        pending.replace(path)


def acquire_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Get the public versioned archive, check the CSV checksum, and validate the rows."""
    cache_path = raw_dir / CACHE_NAME
    cached = cache_path.is_file()
    if cached:
        with cache_path.open("rb") as handle:
            archive = handle.read(MAX_ARCHIVE_BYTES + 1)
    else:
        with urlopen(DOWNLOAD_URL, timeout=45) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpected size for the Kaggle Marketing A/B Testing archive.")

    try:
        with ZipFile(BytesIO(archive)) as source:
            member = source.getinfo(CSV_MEMBER)
            if member.file_size > MAX_CSV_BYTES:
                raise ValueError(f"Unexpected size for Kaggle archive member {CSV_MEMBER}.")
            csv_data = source.read(member)
    except (BadZipFile, KeyError) as error:
        raise ValueError(f"Expected the Kaggle ZIP archive with member {CSV_MEMBER}.") from error

    csv_digest = sha256(csv_data).hexdigest()
    if csv_digest != CSV_SHA256:
        raise ValueError(f"The Kaggle {CSV_MEMBER} SHA-256 does not match the pinned release.")
    data = validate_data(pd.read_csv(BytesIO(csv_data), encoding="utf-8"))
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} Kaggle records, found {len(data):,}.")
    if not cached:
        _cache_archive(cache_path, archive)

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_name": "Marketing A/B Testing",
        "dataset_version": DATASET_VERSION,
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "csv_member": CSV_MEMBER,
        "license": "CC0 1.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "csv_sha256": csv_digest,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "rows": len(data),
    }
    return data, provenance


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns and enforce one outcome and assignment per observed user."""
    normalized = frame.rename(columns=SOURCE_COLUMNS)
    missing = sorted(set(RAW_COLUMNS).difference(normalized.columns))
    extra = sorted(set(normalized.columns).difference(RAW_COLUMNS), key=str)
    if missing or extra or not normalized.columns.is_unique:
        raise ValueError(f"Unexpected Marketing A/B Testing schema. Missing {missing}, extra {extra}.")
    if normalized.empty:
        raise ValueError("Marketing A/B Testing data has no rows.")
    if normalized.isna().any().any():
        null_columns = normalized.columns[normalized.isna().any()].tolist()
        raise ValueError(f"Unexpected missing values in {null_columns}.")

    data = normalized.loc[:, list(RAW_COLUMNS)].copy()
    minimums = {"source_row": 0, "user_id": 1, "total_ads": 0, "most_ads_hour": 0}
    for column, minimum in minimums.items():
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values).all() or values.mod(1).ne(0).any() or values.lt(minimum).any():
            raise ValueError(f"Expected finite integers of at least {minimum} in {column}.")
        data[column] = values.astype("int64")
    if not data["most_ads_hour"].between(0, 23).all():
        raise ValueError("Expected hours from 0 through 23 in most_ads_hour.")

    groups = data["test_group"].astype("string").str.strip().str.lower()
    if set(groups) != set(GROUPS):
        raise ValueError("Expected both ad and psa in test_group, with no other groups.")
    data["test_group"] = groups
    converted = data["converted"].astype("string").str.strip().str.lower().map(BOOLEAN_VALUES)
    if converted.isna().any():
        raise ValueError("Expected true or false in converted.")
    data["converted"] = converted.astype("bool")
    weekdays = data["most_ads_day"].astype("string").str.strip().str.title()
    if not weekdays.isin(WEEKDAYS).all():
        raise ValueError("Expected a weekday name in most_ads_day.")
    data["most_ads_day"] = weekdays

    repeated = data["user_id"].duplicated(keep=False)
    if repeated.any():
        cross_group = data.loc[repeated].groupby("user_id")["test_group"].nunique().gt(1).sum()
        raise ValueError(
            f"Found {int(repeated.sum()):,} rows with repeated user IDs, including "
            f"{int(cross_group):,} IDs assigned to both groups."
        )
    if not data["source_row"].is_unique:
        raise ValueError("Expected the original source-row field to be unique.")

    return data
