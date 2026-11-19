"""Fetch the pinned Criteo research release and load a typed DuckDB projection."""

import gzip
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

DATASET_URL = "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/"
MIRROR_URL = "https://huggingface.co/datasets/criteo/criteo-attribution-dataset"
REVISION = "904188a63cbad78bee43cd26ff5ee4ac77903986"
FILENAME = "criteo_attribution_dataset.tsv.gz"
DOWNLOAD_URL = f"{MIRROR_URL}/resolve/{REVISION}/{FILENAME}"
SHA256 = "94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98"
EXPECTED_SIZE = 653_015_824
EXPECTED_ROWS = 16_468_027
COLUMNS = (
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click",
    *(f"cat{i}" for i in range(1, 10)),
)


def verify_release(path: Path) -> None:
    """Check bytes and header before reading the full dataset."""
    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected Criteo file size: {path}")
    with path.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected Criteo SHA-256: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        header = tuple(file.readline().rstrip("\r\n").split("\t"))
    if header != COLUMNS:
        raise ValueError(f"Unexpected Criteo header: {header}")


def acquire_data(raw_dir: Path, local_file: Path | None = None) -> Path:
    """Use the pinned local file or cache an authenticated-by-digest download."""
    if local_file is not None:
        verify_release(local_file)
        return local_file

    path = raw_dir / FILENAME
    if path.exists():
        verify_release(path)
        return path

    raw_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=raw_dir, suffix=".part", delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            with urlopen(DOWNLOAD_URL, timeout=60) as response:
                copyfileobj(response, temporary, length=1024 * 1024)
            temporary.close()
            verify_release(pending)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return path


def load_impressions(connection, path: Path) -> None:
    """Store only the numeric fields used in the quality and path audit."""
    connection.execute(
        """
        CREATE TEMP TABLE impressions AS
        SELECT timestamp::BIGINT AS touch_time, uid::BIGINT AS user_id,
               campaign::BIGINT AS campaign, conversion::TINYINT AS has_conversion,
               conversion_timestamp::BIGINT AS conversion_time,
               conversion_id::BIGINT AS conversion_id, attribution::TINYINT AS attributed
        FROM read_csv(?, delim = '\t', header = true, compression = 'gzip', all_varchar = true)
        """,
        [str(path)],
    )
    count = connection.execute("SELECT count(*) FROM impressions").fetchone()[0]
    if count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} Criteo impressions, found {count:,}.")
