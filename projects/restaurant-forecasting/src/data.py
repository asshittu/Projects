"""Load authorized, local competition files without publishing raw records."""

import logging
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from zipfile import BadZipFile, ZipFile, ZipInfo
from zlib import error as DecompressionError

import numpy as np
import pandas as pd

INPUT_FIELDS = {
    "air_visit_data.csv": ("air_store_id", "visit_date", "visitors"),
    "date_info.csv": ("calendar_date", "holiday_flg"),
    "air_reserve.csv": ("air_store_id", "visit_datetime", "reserve_datetime", "reserve_visitors"),
    "sample_submission.csv": ("id", "visitors"),
}
MAX_CSV_BYTES = 128 * 1024**2
MAX_NESTED_ZIP_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 256 * 1024**2
LOGGER = logging.getLogger(__name__)


class _InputSource(NamedTuple):
    path: Path
    outer_member: ZipInfo | None = None
    inner_member: ZipInfo | None = None


def find_competition_directory(input_root: Path | str) -> Path:
    """Find the sole directory containing all four loose CSVs."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError("The competition input root must be an accessible directory.")
    matches = {
        path.parent
        for path in root.rglob("air_visit_data.csv")
        if all((path.parent / filename).is_file() for filename in INPUT_FIELDS)
    }
    if not matches:
        raise FileNotFoundError("No complete competition CSV directory was found under the input root.")
    if len(matches) > 1:
        raise ValueError("Multiple complete CSV directories were found. Select one competition directory.")
    return matches.pop()


def _member_name(member: ZipInfo) -> str:
    return PurePosixPath(member.filename.replace("\\", "/")).name


def _read_zip_member(archive: ZipFile, member: ZipInfo, limit: int, name: str) -> bytes:
    if member.file_size > limit:
        raise ValueError(f"A ZIP member for {name} exceeds the supported read size.")
    with archive.open(member) as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"A ZIP member for {name} exceeds the supported read size.")
    return content


def _add_zip_sources(path: Path, matches: dict[str, list[_InputSource]], *, direct: bool) -> None:
    wrapper = path.name.removesuffix(".zip")
    expected = wrapper if wrapper in INPUT_FIELDS else None
    label = expected or "the competition CSVs"
    try:
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            if direct or expected:
                raise ValueError(f"A ZIP input for {label} exceeds the supported read size.")
            return
        archive = ZipFile(path)
    except (OSError, BadZipFile):
        if direct or expected:
            raise ValueError(f"Expected a readable ZIP input for {label}.") from None
        return

    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            basename = _member_name(member)
            if basename in INPUT_FIELDS and expected in (None, basename):
                matches[basename].append(_InputSource(path, member))
                continue
            nested_name = basename.removesuffix(".zip")
            if expected or basename == nested_name or nested_name not in INPUT_FIELDS:
                continue
            try:
                packed = _read_zip_member(archive, member, MAX_NESTED_ZIP_BYTES, nested_name)
                with ZipFile(BytesIO(packed)) as inner:
                    for entry in inner.infolist():
                        if not entry.is_dir() and _member_name(entry) == nested_name:
                            matches[nested_name].append(_InputSource(path, member, entry))
            except (BadZipFile, OSError, RuntimeError, NotImplementedError, DecompressionError):
                raise ValueError(f"Expected a readable nested ZIP input for {nested_name}.") from None


def _resolve_inputs(input_root: Path | str) -> dict[str, _InputSource]:
    root = Path(input_root)
    matches: dict[str, list[_InputSource]] = {name: [] for name in INPUT_FIELDS}
    if root.is_file() and root.suffix.lower() == ".zip":
        _add_zip_sources(root, matches, direct=True)
    elif root.is_dir():
        for path in root.rglob("*"):
            if path.name in INPUT_FIELDS and path.is_file():
                matches[path.name].append(_InputSource(path))
            elif path.suffix.lower() == ".zip" and path.is_file():
                _add_zip_sources(path, matches, direct=False)
    else:
        raise FileNotFoundError("Select an accessible directory or ZIP containing the competition inputs.")

    missing = [name for name, sources in matches.items() if not sources]
    ambiguous = [name for name, sources in matches.items() if len(sources) > 1]
    missing_message = f"Missing required files: {', '.join(missing)}." if missing else ""
    if ambiguous:
        problem = f"Multiple copies found for {', '.join(ambiguous)}. Select an input with one copy of each."
        raise ValueError(f"{missing_message} {problem}".strip())
    if missing:
        raise FileNotFoundError(missing_message)
    return {name: sources[0] for name, sources in matches.items()}


def _read_content(source: _InputSource, name: str) -> bytes:
    try:
        if source.outer_member is None:
            with source.path.open("rb") as handle:
                content = handle.read(MAX_CSV_BYTES + 1)
            if len(content) > MAX_CSV_BYTES:
                raise ValueError(f"The input for {name} exceeds the supported read size.")
            return content
        with ZipFile(source.path) as archive:
            if source.inner_member is None:
                return _read_zip_member(archive, source.outer_member, MAX_CSV_BYTES, name)
            packed = _read_zip_member(archive, source.outer_member, MAX_NESTED_ZIP_BYTES, name)
            with ZipFile(BytesIO(packed)) as inner:
                return _read_zip_member(inner, source.inner_member, MAX_CSV_BYTES, name)
    except (BadZipFile, OSError, RuntimeError, NotImplementedError, DecompressionError):
        raise ValueError(f"Could not read the resolved competition input for {name}.") from None


def _read_input(source: _InputSource, name: str) -> pd.DataFrame:
    columns = list(INPUT_FIELDS[name])
    content = _read_content(source, name)
    try:
        frame = pd.read_csv(BytesIO(content), usecols=columns, dtype="string", keep_default_na=False)
    except (OSError, ValueError):
        raise ValueError(f"Expected a readable {name} with columns {', '.join(columns)}.") from None
    if frame.empty and name != "air_reserve.csv":
        raise ValueError(f"Expected at least one row in {name}.")
    return frame.loc[:, columns]


def _identifiers(values: pd.Series, field: str) -> pd.Series:
    identifiers = values.str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError(f"Expected nonblank identifiers in {field}.")
    return identifiers


def _counts(values: pd.Series, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric).all()
        or numeric.lt(0).any()
        or numeric.ge(2**63).any()
        or numeric.mod(1).ne(0).any()
    ):
        raise ValueError(f"Expected finite nonnegative integer counts in {field}.")
    return numeric.astype("int64")


def _dates(values: pd.Series, field: str, *, date_only: bool) -> pd.Series:
    date_format = "%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M:%S"
    parsed = pd.to_datetime(values, format=date_format, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Expected valid timezone-naive dates or timestamps in {field}.")
    return parsed.astype("datetime64[ns]")


def _scheduled_dates(sample: pd.DataFrame) -> pd.DatetimeIndex:
    ids = _identifiers(sample["id"], "sample_submission.csv id")
    if not ids.is_unique:
        raise ValueError("Expected unique IDs in sample_submission.csv.")
    if not ids.str.match(r"^.+_\d{4}-\d{2}-\d{2}$").all():
        raise ValueError("Expected sample_submission.csv IDs to end with _YYYY-MM-DD.")
    _counts(sample["visitors"], "sample_submission.csv visitors")
    date_suffix = _dates(ids.str[-10:], "sample_submission.csv ID date suffix", date_only=True)
    horizon = pd.DatetimeIndex(date_suffix.unique()).sort_values()
    if not horizon.equals(pd.date_range(horizon[0], periods=len(horizon), freq="D")):
        raise ValueError("The dates in sample_submission.csv must define a continuous daily horizon.")
    return horizon


def load_competition_data(
    input_root: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Read a directory or ZIP and return validated data and the scheduled horizon length."""
    sources = _resolve_inputs(input_root)
    visits = _read_input(sources["air_visit_data.csv"], "air_visit_data.csv")
    visits["air_store_id"] = _identifiers(visits["air_store_id"], "air_visit_data.csv air_store_id")
    visits["visit_date"] = _dates(visits["visit_date"], "visit_date", date_only=True)
    visits["visitors"] = _counts(visits["visitors"], "air_visit_data.csv visitors")
    if visits.duplicated(subset=["air_store_id", "visit_date"]).any():
        raise ValueError("Expected one recorded visit count per AIR store and date.")

    calendar = _read_input(sources["date_info.csv"], "date_info.csv").rename(columns={"calendar_date": "visit_date"})
    calendar["visit_date"] = _dates(calendar["visit_date"], "calendar_date", date_only=True)
    calendar["holiday_flg"] = _counts(calendar["holiday_flg"], "holiday_flg")
    if not calendar["visit_date"].is_unique or not calendar["holiday_flg"].isin((0, 1)).all():
        raise ValueError("Expected one calendar row per date and holiday flags of 0 or 1.")

    reservations = _read_input(sources["air_reserve.csv"], "air_reserve.csv")
    reservations["air_store_id"] = _identifiers(reservations["air_store_id"], "air_reserve.csv air_store_id")
    for field in ("visit_datetime", "reserve_datetime"):
        reservations[field] = _dates(reservations[field], field, date_only=False)
    reservations["reserve_visitors"] = _counts(reservations["reserve_visitors"], "reserve_visitors")
    late = reservations["reserve_datetime"].gt(reservations["visit_datetime"])
    if late.any():
        LOGGER.warning("Ignored post-visit reservation records (count %d).", int(late.sum()))
        reservations = reservations.loc[~late].copy()
    reservations["visit_date"] = reservations["visit_datetime"].dt.normalize()

    horizon = _scheduled_dates(_read_input(sources["sample_submission.csv"], "sample_submission.csv"))
    if horizon[0] <= visits["visit_date"].max():
        raise ValueError("Expected the scheduled horizon to follow all recorded historical visits.")
    required_dates = pd.DatetimeIndex(visits["visit_date"].unique()).union(horizon)
    if not required_dates.difference(calendar["visit_date"]).empty:
        raise ValueError("Expected calendar coverage for all historical visit dates and the scheduled horizon.")

    return visits, calendar, reservations, len(horizon)
