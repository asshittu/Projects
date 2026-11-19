"""Load authorized, locally mounted competition CSVs without publishing raw records."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FIELDS = {
    "air_visit_data.csv": ("air_store_id", "visit_date", "visitors"),
    "date_info.csv": ("calendar_date", "holiday_flg"),
    "air_reserve.csv": ("air_store_id", "visit_datetime", "reserve_datetime", "reserve_visitors"),
    "sample_submission.csv": ("id", "visitors"),
}
LOGGER = logging.getLogger(__name__)


def find_competition_directory(input_root: Path | str) -> Path:
    """Find the sole directory containing all four expected CSVs."""
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


def _read_input(directory: Path, name: str) -> pd.DataFrame:
    columns = list(INPUT_FIELDS[name])
    try:
        frame = pd.read_csv(directory / name, usecols=columns, dtype="string", keep_default_na=False)
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
    directory: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Return validated visits, calendar, reservations, and the length of the scheduled horizon."""
    root = Path(directory)
    visits = _read_input(root, "air_visit_data.csv")
    visits["air_store_id"] = _identifiers(visits["air_store_id"], "air_visit_data.csv air_store_id")
    visits["visit_date"] = _dates(visits["visit_date"], "visit_date", date_only=True)
    visits["visitors"] = _counts(visits["visitors"], "air_visit_data.csv visitors")
    if visits.duplicated(subset=["air_store_id", "visit_date"]).any():
        raise ValueError("Expected one recorded visit count per AIR store and date.")

    calendar = _read_input(root, "date_info.csv").rename(columns={"calendar_date": "visit_date"})
    calendar["visit_date"] = _dates(calendar["visit_date"], "calendar_date", date_only=True)
    calendar["holiday_flg"] = _counts(calendar["holiday_flg"], "holiday_flg")
    if not calendar["visit_date"].is_unique or not calendar["holiday_flg"].isin((0, 1)).all():
        raise ValueError("Expected one calendar row per date and holiday flags of 0 or 1.")

    reservations = _read_input(root, "air_reserve.csv")
    reservations["air_store_id"] = _identifiers(reservations["air_store_id"], "air_reserve.csv air_store_id")
    for field in ("visit_datetime", "reserve_datetime"):
        reservations[field] = _dates(reservations[field], field, date_only=False)
    reservations["reserve_visitors"] = _counts(reservations["reserve_visitors"], "reserve_visitors")
    late = reservations["reserve_datetime"].gt(reservations["visit_datetime"])
    if late.any():
        LOGGER.warning("Ignored post-visit reservation records (count %d).", int(late.sum()))
        reservations = reservations.loc[~late].copy()
    reservations["visit_date"] = reservations["visit_datetime"].dt.normalize()

    horizon = _scheduled_dates(_read_input(root, "sample_submission.csv"))
    if horizon[0] <= visits["visit_date"].max():
        raise ValueError("Expected the scheduled horizon to follow all recorded historical visits.")
    required_dates = pd.DatetimeIndex(visits["visit_date"].unique()).union(horizon)
    if not required_dates.difference(calendar["visit_date"]).empty:
        raise ValueError("Expected calendar coverage for all historical visit dates and the scheduled horizon.")

    return visits, calendar, reservations, len(horizon)
