"""Acquire the Yelp subset of the UCI Sentiment Labelled Sentences dataset."""

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
CACHE_NAME = "sentiment-labelled-sentences.zip"
SOURCE_MEMBER = "sentiment labelled sentences/yelp_labelled.txt"
SOURCE_SHA256 = "c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea"
EXPECTED_ROWS = 1_000
MAX_ARCHIVE_BYTES = 1024**2
MAX_SOURCE_BYTES = 512 * 1024
COLUMNS = ("text", "label")


def _cache_archive(path: Path, archive: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(archive)
        pending.replace(path)


def _read_reviews(source: bytes) -> pd.DataFrame:
    rows = []
    for line_number, line in enumerate(source.decode("utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2 or fields[1].strip() not in ("0", "1"):
            raise ValueError(f"Invalid review and binary label on UCI line {line_number}.")
        rows.append((fields[0], int(fields[1])))
    return pd.DataFrame(rows, columns=COLUMNS)


def _label_counts(data: pd.DataFrame) -> dict[str, int]:
    return {str(label): int(data["label"].eq(label).sum()) for label in (0, 1)}


def acquire_data(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Verify the original Yelp file, validate labels, and remove repeated text profiles."""
    cache_path = raw_dir / CACHE_NAME
    cached = cache_path.is_file()
    if cached:
        with cache_path.open("rb") as handle:
            archive = handle.read(MAX_ARCHIVE_BYTES + 1)
    else:
        with urlopen(DOWNLOAD_URL, timeout=30) as response:
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("Unexpected size for the UCI Sentiment Labelled Sentences archive.")

    try:
        with ZipFile(BytesIO(archive)) as source:
            member = source.getinfo(SOURCE_MEMBER)
            if member.file_size > MAX_SOURCE_BYTES:
                raise ValueError(f"Unexpected size for UCI archive member {SOURCE_MEMBER}.")
            reviews = source.read(member)
    except (BadZipFile, KeyError) as error:
        raise ValueError(f"Expected the UCI ZIP archive with member {SOURCE_MEMBER}.") from error

    source_digest = sha256(reviews).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError("The Yelp subset SHA-256 does not match the pinned release.")
    raw = _read_reviews(reviews)
    if len(raw) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} Yelp rows, found {len(raw):,}.")
    data = validate_data(raw)
    if not cached:
        _cache_archive(cache_path, archive)

    provenance = {
        "dataset_id": 331,
        "dataset_name": "UCI Sentiment Labelled Sentences (Yelp subset)",
        "dataset_url": DATASET_URL,
        "download_url": DOWNLOAD_URL,
        "source_member": SOURCE_MEMBER,
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "source_sha256": source_digest,
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "raw_rows": len(raw),
        "retained_rows": len(data),
        "duplicates_removed": len(raw) - len(data),
        "raw_label_counts": _label_counts(raw),
        "retained_label_counts": _label_counts(data),
        "rows": len(data),
    }
    return data, provenance


def text_keys(text: pd.Series) -> pd.Series:
    """Return normalized comparison keys without rewriting the model's source text."""
    return text.str.normalize("NFKC").str.split().str.join(" ").str.casefold()


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep first source representatives of unique, normalized review text profiles."""
    if set(frame.columns) != set(COLUMNS) or not frame.columns.is_unique or frame.empty:
        raise ValueError("Expected nonempty review data with text and label columns.")
    if frame.isna().any().any():
        raise ValueError("Expected review text and labels without missing values.")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Expected strings in text.")
    label = pd.to_numeric(frame["label"], errors="coerce")
    if not label.isin((0, 1)).all():
        raise ValueError("Expected binary 0/1 labels.")

    text = frame["text"].astype("string")
    key = text_keys(text)
    if key.eq("").any():
        raise ValueError("Expected nonblank review text.")
    if label.groupby(key, sort=False).nunique().gt(1).any():
        raise ValueError("A normalized review text profile has conflicting labels.")

    clean = pd.DataFrame({"text": text, "label": label.astype("int8")})
    return clean.loc[~key.duplicated()].reset_index(drop=True)
