"""Leakage-safe AIR visit backtests and aggregate planning scenarios."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

DAY = pd.Timedelta(days=1)
STORE_DATE = ["air_store_id", "visit_date"]
FEATURES = [
    "lead_days",
    "weekday",
    "month",
    "holiday_flg",
    "seasonal_naive",
    "store_mean",
    "store_history_days",
    "weekday_mean",
    "recent_28d_mean",
    "booked_reservations",
    "booked_visitors",
]
CANDIDATES = ("seasonal_naive", "hist_gradient_boosting")


def feature_snapshot(
    visits: pd.DataFrame,
    targets: pd.DataFrame,
    calendar: pd.DataFrame,
    reservations: pd.DataFrame,
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Create target-day features from a snapshot taken at the end of the origin date."""
    history = visits.loc[visits["visit_date"] <= origin].copy()
    if history.empty:
        raise ValueError("Forecast origins require earlier observed visits")

    features = targets.loc[:, STORE_DATE].copy()
    if (features["visit_date"] <= origin).any():
        raise ValueError("Target dates must be strictly after the forecast origin")
    features["lead_days"] = (features["visit_date"] - origin).dt.days
    features["weekday"] = features["visit_date"].dt.dayofweek
    features["month"] = features["visit_date"].dt.month
    features = features.merge(calendar[["visit_date", "holiday_flg"]], on="visit_date", how="left", validate="many_to_one")
    if features["holiday_flg"].isna().any():
        raise ValueError("Published calendar does not cover every target date")

    store = history.groupby("air_store_id").agg(
        store_mean=("visitors", "mean"),
        store_history_days=("visitors", "size"),
    )
    history["weekday"] = history["visit_date"].dt.dayofweek
    weekday = (
        history.sort_values("visit_date", kind="stable")
        .groupby(["air_store_id", "weekday"])
        .agg(weekday_mean=("visitors", "mean"), last_weekday_visits=("visitors", "last"))
    )
    recent = (
        history.loc[history["visit_date"] > origin - 28 * DAY]
        .groupby("air_store_id")["visitors"].mean().rename("recent_28d_mean")
    )
    features = features.join(store, on="air_store_id", validate="many_to_one")
    features = features.join(weekday, on=["air_store_id", "weekday"], validate="many_to_one")
    features = features.join(recent, on="air_store_id", validate="many_to_one")
    features["seasonal_naive"] = (
        features["last_weekday_visits"].fillna(features["store_mean"]).fillna(history["visitors"].mean())
    )
    features["store_history_days"] = features["store_history_days"].fillna(0)

    # Midnight after origin is excluded, even if the reservation precedes its visit.
    booked = reservations.loc[
        (reservations["reserve_datetime"] < origin + DAY)
        & (reservations["visit_date"] > origin)
        & reservations["visit_date"].isin(features["visit_date"])
    ]
    reservation_totals = booked.groupby(STORE_DATE).agg(
        booked_reservations=("reserve_visitors", "size"),
        booked_visitors=("reserve_visitors", "sum"),
    )
    features = features.join(reservation_totals, on=STORE_DATE, validate="many_to_one")
    features[["booked_reservations", "booked_visitors"]] = features[["booked_reservations", "booked_visitors"]].fillna(0)
    return features[FEATURES]


def target_window(visits: pd.DataFrame, origin: pd.Timestamp, horizon: pd.Timedelta) -> pd.DataFrame:
    """Select only observed store-days in the forecast horizon."""
    return visits.loc[(visits["visit_date"] > origin) & (visits["visit_date"] <= origin + horizon)]


def training_snapshots(
    visits: pd.DataFrame,
    calendar: pd.DataFrame,
    reservations: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    horizon: pd.Timedelta,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Replay consecutive, non-overlapping horizons with all labels known by origin."""
    oldest_visit = visits["visit_date"].min()
    historical_origins = []
    simulated_origin = origin - horizon
    while simulated_origin >= oldest_visit:
        historical_origins.append(simulated_origin)
        simulated_origin -= horizon

    predictors = []
    labels = []
    for simulated_origin in reversed(historical_origins):
        targets = target_window(visits, simulated_origin, horizon)
        if targets.empty:
            continue
        predictors.append(feature_snapshot(visits, targets, calendar, reservations, origin=simulated_origin))
        labels.append(targets["visitors"].to_numpy(dtype=float))

    if not predictors:
        raise ValueError("Not enough completed historical forecast windows to train the model")
    return pd.concat(predictors, ignore_index=True), np.concatenate(labels), len(predictors)


def error_totals(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Return additive absolute and squared log errors for exact pooling."""
    absolute = np.abs(forecast - actual).sum()
    squared_log = np.square(np.log1p(forecast) - np.log1p(actual)).sum()
    return float(absolute), float(squared_log)


def summarize_errors(totals: dict, n: int) -> dict:
    """Convert pooled loss totals to aggregate MAE and RMSLE."""
    return {
        "n_evaluated": n,
        "models": {
            key: {"mae": totals[key][0] / n, "rmsle": float(np.sqrt(totals[key][1] / n))}
            for key in CANDIDATES
        },
    }


def planning_scenario(actual: np.ndarray, forecast: np.ndarray, model: str) -> dict:
    """Estimate illustrative capacity with 18 visitors per shift and 10% headroom."""
    visitors_per_shift = 18
    headroom = 0.10
    without_headroom = np.ceil(forecast / visitors_per_shift).astype(np.int64)
    with_headroom = np.ceil(forecast * (1 + headroom) / visitors_per_shift).astype(np.int64)
    return {
        "model": model,
        "n_evaluated": len(actual),
        "visitors_per_shift": visitors_per_shift,
        "headroom": headroom,
        "shift_slots_without_headroom": int(without_headroom.sum()),
        "shift_slots_with_headroom": int(with_headroom.sum()),
        "under_capacity_days_without_headroom": int((actual > without_headroom * visitors_per_shift).sum()),
        "under_capacity_days_with_headroom": int((actual > with_headroom * visitors_per_shift).sum()),
    }


def evaluate_forecasts(
    visits: pd.DataFrame,
    calendar: pd.DataFrame,
    reservations: pd.DataFrame,
    *,
    horizon_days: int,
    n_folds: int = 3,
) -> dict:
    """Fit and test at rolling origins, returning only aggregate per-window results."""
    if not isinstance(horizon_days, (int, np.integer)) or horizon_days <= 0 or n_folds < 2:
        raise ValueError("A positive day horizon and at least two folds are required")
    if visits.empty:
        raise ValueError("Observed visits are required for the backtest")

    horizon = int(horizon_days) * DAY
    latest_date = visits["visit_date"].max()
    folds = []
    totals = {key: np.zeros(2) for key in CANDIDATES}
    n_overall = 0
    for index in range(n_folds):
        holdout = index == n_folds - 1
        if holdout:
            validation = summarize_errors(totals, n_overall)
            selected = min(CANDIDATES, key=lambda key: validation["models"][key]["rmsle"])

        origin = latest_date - (n_folds - index) * horizon
        targets = target_window(visits, origin, horizon)
        if targets.empty:
            raise ValueError(f"No target observations in window after {origin.date()}")
        train, labels, n_simulated_origins = training_snapshots(
            visits, calendar, reservations, origin=origin, horizon=horizon,
        )
        future = feature_snapshot(visits, targets, calendar, reservations, origin=origin)
        model = HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            learning_rate=0.07,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
        with threadpool_limits(limits=1):
            model.fit(train, labels)
            prediction = np.maximum(model.predict(future), 0)
        forecasts = {"seasonal_naive": future["seasonal_naive"].to_numpy(dtype=float), "hist_gradient_boosting": prediction}
        actual = targets["visitors"].to_numpy(dtype=float)
        losses = {key: error_totals(actual, forecast) for key, forecast in forecasts.items()}
        folds.append({
            "name": "holdout" if holdout else f"validation_{index + 1}",
            "origin": origin.date().isoformat(),
            "target_start": (origin + DAY).date().isoformat(),
            "target_end": (origin + horizon).date().isoformat(),
            "n_train": len(labels),
            "n_simulated_training_origins": n_simulated_origins,
            **summarize_errors(losses, len(actual)),
        })
        for key in CANDIDATES:
            totals[key] += losses[key]
        n_overall += len(actual)

        if holdout:
            planning = planning_scenario(actual, forecasts[selected], selected)

    return {
        "horizon_days": int(horizon_days),
        "n_folds": n_folds,
        "folds": folds,
        "validation": validation,
        "overall": summarize_errors(totals, n_overall),
        "selection": {"model": selected, "criterion": "pooled_validation_rmsle", "tie_breaker": "seasonal_naive"},
        "planning": planning,
    }
