"""Verify origin cutoffs, fallbacks, and aggregate backtests with artificial data."""

import json
import unittest

import numpy as np
import pandas as pd

from src.forecast import (
    evaluate_forecasts,
    feature_snapshot,
    planning_scenario,
    training_snapshots,
)


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two fictitious stores with weekday and slowly changing visit counts."""
    dates = pd.date_range("2020-01-01", periods=84, freq="D")
    visits = pd.DataFrame([
        (store, date, base + 2 * date.dayofweek + (day // 14))
        for store, base in (("synthetic_a", 10), ("synthetic_b", 30))
        for day, date in enumerate(dates)
    ], columns=["air_store_id", "visit_date", "visitors"])
    calendar = pd.DataFrame({"visit_date": dates, "holiday_flg": (dates.dayofweek == 6).astype(int)})
    reservations = pd.DataFrame({
        "air_store_id": ["synthetic_a"] * len(dates),
        "visit_date": dates,
        "reserve_datetime": dates - pd.Timedelta(days=2) + pd.Timedelta(hours=11),
        "reserve_visitors": [4] * len(dates),
    })
    return visits, calendar, reservations


class SnapshotTests(unittest.TestCase):
    def test_snapshot_excludes_future_visits_and_midnight_reservations(self):
        origin = pd.Timestamp("2020-01-14")
        target = pd.Timestamp("2020-01-15")
        visits = pd.DataFrame([
            ("synthetic_a", pd.Timestamp("2020-01-01"), 10),
            ("synthetic_a", pd.Timestamp("2020-01-08"), 20),
            ("synthetic_b", pd.Timestamp("2020-01-09"), 30),
            ("synthetic_a", target, 100),
            ("synthetic_b", target, 100),
            ("synthetic_c", target, 100),
        ], columns=["air_store_id", "visit_date", "visitors"])
        targets = visits.loc[visits["visit_date"] == target]
        calendar = pd.DataFrame({"visit_date": [target], "holiday_flg": [1]})
        reservations = pd.DataFrame({
            "air_store_id": ["synthetic_a"] * 3,
            "visit_date": [target] * 3,
            "reserve_datetime": pd.to_datetime(["2020-01-13 12:00", "2020-01-14 23:59", "2020-01-15 00:00"]),
            "reserve_visitors": [4, 5, 100],
        })

        features = feature_snapshot(visits, targets, calendar, reservations, origin=origin)
        self.assertEqual(features["seasonal_naive"].tolist(), [20, 30, 20])
        self.assertEqual(features["booked_visitors"].tolist(), [9, 0, 0])
        self.assertEqual(features["booked_reservations"].tolist(), [2, 0, 0])
        self.assertEqual(features["store_history_days"].tolist(), [2, 1, 0])
        self.assertEqual(features["holiday_flg"].tolist(), [1, 1, 1])
        self.assertNotIn("visitors", features.columns)

        perturbed_visits = visits.copy()
        perturbed_visits.loc[perturbed_visits["visit_date"] > origin, "visitors"] *= 1_000
        perturbed_reservations = reservations.copy()
        perturbed_reservations.loc[2, "reserve_visitors"] *= 1_000
        actual = feature_snapshot(perturbed_visits, targets, calendar, perturbed_reservations, origin=origin)
        pd.testing.assert_frame_equal(features, actual)

    def test_training_labels_and_features_are_invariant_to_post_origin_visits(self):
        visits, calendar, reservations = fixture()
        origin = pd.Timestamp("2020-02-26")
        horizon = pd.Timedelta(days=7)
        before = training_snapshots(visits, calendar, reservations, origin=origin, horizon=horizon)
        changed = visits.copy()
        changed.loc[changed["visit_date"] > origin, "visitors"] = 10_000
        after = training_snapshots(changed, calendar, reservations, origin=origin, horizon=horizon)

        self.assertEqual(before[2], 8)
        self.assertEqual(len(before[1]), 112)
        self.assertEqual(set(before[0]["lead_days"]), set(range(1, 8)))
        pd.testing.assert_frame_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_requires_calendar_date_coverage(self):
        visits, calendar, reservations = fixture()
        target = pd.Timestamp("2020-01-15")
        targets = visits.loc[visits["visit_date"] == target]
        missing = calendar.loc[calendar["visit_date"] != target]
        with self.assertRaisesRegex(ValueError, "calendar does not cover"):
            feature_snapshot(visits, targets, missing, reservations, origin=target - pd.Timedelta(days=1))


class EvaluationTests(unittest.TestCase):
    def test_backtest_windows_are_disjoint_and_outputs_are_aggregate(self):
        visits, calendar, reservations = fixture()
        result = evaluate_forecasts(visits, calendar, reservations, horizon_days=7)
        self.assertEqual([item["name"] for item in result["folds"]], ["validation_1", "validation_2", "holdout"])
        self.assertEqual(result["overall"]["n_evaluated"], 42)
        self.assertEqual(result["validation"]["n_evaluated"], 28)
        self.assertEqual(result["folds"][-1]["target_end"], visits["visit_date"].max().date().isoformat())
        for left, right in zip(result["folds"], result["folds"][1:]):
            self.assertEqual(left["target_end"], right["origin"])
            self.assertLess(left["target_end"], right["target_start"])
        self.assertEqual(result["planning"]["model"], result["selection"]["model"])
        self.assertEqual(result["planning"]["n_evaluated"], 14)
        self.assertGreaterEqual(result["planning"]["shift_slots_with_headroom"], result["planning"]["shift_slots_without_headroom"])
        self.assertNotIn("synthetic_a", json.dumps(result, allow_nan=False))
        self.assertNotIn("synthetic_b", json.dumps(result, allow_nan=False))

        for model, validation in result["validation"]["models"].items():
            validation_folds = result["folds"][:2]
            mae = sum(fold["models"][model]["mae"] * fold["n_evaluated"] for fold in validation_folds) / 28
            rmsle = np.sqrt(sum(fold["models"][model]["rmsle"] ** 2 * fold["n_evaluated"] for fold in validation_folds) / 28)
            self.assertAlmostEqual(validation["mae"], mae)
            self.assertAlmostEqual(validation["rmsle"], rmsle)

    def test_hypothetical_shift_capacity_assumptions_are_explicit(self):
        result = planning_scenario(np.array([19.0, 20.0, 1.0]), np.array([18.0, 18.0, 0.0]), "seasonal_naive")
        self.assertEqual(result["visitors_per_shift"], 18)
        self.assertEqual(result["headroom"], 0.10)
        self.assertEqual(result["shift_slots_without_headroom"], 2)
        self.assertEqual(result["shift_slots_with_headroom"], 4)
        self.assertEqual(result["under_capacity_days_without_headroom"], 3)
        self.assertEqual(result["under_capacity_days_with_headroom"], 1)


if __name__ == "__main__":
    unittest.main()
