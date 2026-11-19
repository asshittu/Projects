"""Small fixtures verify attribution calculations without downloading source data."""

import unittest

import duckdb

from src.analysis import analyze, source_summary


DAY = 86_400


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.connection = duckdb.connect()
        self.addCleanup(self.connection.close)
        self.connection.execute("""
            CREATE TEMP TABLE impressions (
                touch_time BIGINT, user_id BIGINT, campaign BIGINT, has_conversion INTEGER,
                conversion_time BIGINT, conversion_id BIGINT, attributed INTEGER
            )
        """)

    def load(self, rows):
        self.connection.executemany("INSERT INTO impressions VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def test_time_limits_ties_and_credit_conservation(self):
        self.load([
            (0, 1, 90, 0, -1, -1, 0),
            (2 * DAY - 1, 1, 90, 1, 3 * DAY, 101, 0),
            (2 * DAY, 1, 11, 1, 3 * DAY, 101, 0),
            (2 * DAY, 1, 22, 1, 3 * DAY, 101, 0),
            (2 * DAY + 1, 1, 33, 1, 3 * DAY, 101, 1),
            (3 * DAY, 1, 90, 1, 3 * DAY, 101, 1),
            (2 * DAY + 30, 2, 33, 1, 3 * DAY + 30, 102, 0),
            (1, 3, 90, 1, DAY // 2, 103, 0),
            (5 * DAY, 4, 90, 1, 6 * DAY, 104, 0),
            (2 * DAY, 5, 90, 1, 3 * DAY, 105, 0),
            (2 * DAY, 6, 90, 1, 3 * DAY, 105, 0),
            (DAY, 7, 90, 1, 3 * DAY, 106, 0),
        ])
        metrics, results, _ = analyze(self.connection, 1)
        self.assertEqual(metrics["paths"], {
            "reported_conversion_ids": 6, "ambiguous_conversion_ids": 1,
            "insufficient_prehistory_ids": 1, "beyond_observation_ids": 1,
            "within_observation_ids": 3, "no_touch_in_lookback_ids": 1,
            "qualified_conversions": 2, "qualified_impressions": 4,
            "multi_touch_conversions": 1, "multi_campaign_conversions": 1,
            "timestamp_tie_conversions": 1,
        })
        self.assertEqual(metrics["source"]["at_or_after_conversion_impressions"], 1)
        self.assertEqual(metrics["credit_totals"], {
            "first_touch": 2, "last_touch": 2, "linear_touch": 2,
        })
        credit = {row["campaign"]: row for row in results}
        self.assertEqual(set(credit), {11, 22, 33})
        self.assertAlmostEqual(credit[11]["first_touch"], 0.5)
        self.assertAlmostEqual(credit[22]["first_touch"], 0.5)
        self.assertAlmostEqual(credit[33]["last_touch"], 2)
        self.assertAlmostEqual(credit[33]["linear_touch"], 4 / 3)

    def test_shared_last_touch_across_repeated_campaign(self):
        self.load([
            (0, 1, 99, 0, -1, -1, 0),
            (DAY, 2, 11, 1, 2 * DAY, 1, 0),
            (DAY + 1, 2, 11, 1, 2 * DAY, 1, 0),
            (DAY + 1, 2, 22, 1, 2 * DAY, 1, 0),
            (2 * DAY, 1, 99, 0, -1, -1, 0),
        ])
        _, results, _ = analyze(self.connection, 1)
        credit = {row["campaign"]: row for row in results}
        self.assertEqual(credit[11]["paths"], 1)
        self.assertEqual(credit[11]["touches"], 2)
        self.assertAlmostEqual(credit[11]["last_touch"], 0.5)
        self.assertAlmostEqual(credit[11]["linear_touch"], 2 / 3)
        self.assertAlmostEqual(credit[22]["last_touch"], 0.5)

    def test_same_user_paths_clip_at_previous_conversion(self):
        self.load([
            (0, 1, 90, 0, -1, -1, 0),
            (DAY + 50, 2, 11, 1, 2 * DAY, 101, 0),
            (DAY + 200, 2, 22, 1, 2 * DAY + 120, 102, 0),
            (DAY + 250, 2, 99, 0, -1, -1, 0),
            (2 * DAY, 2, 99, 0, -1, -1, 0),
            (2 * DAY + 10, 2, 33, 0, -1, -1, 0),
            (2 * DAY + 60, 2, 22, 1, 2 * DAY + 120, 102, 0),
            (2 * DAY + 90, 3, 77, 0, -1, -1, 0),
            (3 * DAY, 1, 90, 0, -1, -1, 0),
        ])
        metrics, _, sensitivity = analyze(self.connection, 1)
        self.assertEqual(metrics["paths"]["multi_campaign_conversions"], 0)
        nearby_paths = metrics["same_user_sensitivity"]["paths"]
        self.assertEqual(nearby_paths["qualified_conversions"], 2)
        self.assertEqual(nearby_paths["multi_campaign_conversions"], 2)
        self.assertEqual(nearby_paths["qualified_impressions"], 5)
        credit = {row["campaign"]: row for row in sensitivity}
        self.assertEqual(set(credit), {11, 22, 33, 99})
        self.assertEqual(credit[99]["touches"], 1)
        self.assertAlmostEqual(credit[99]["last_touch"], 1)
        self.assertAlmostEqual(credit[33]["first_touch"], 1)
        self.assertAlmostEqual(credit[22]["last_touch"], 1)

    def test_inconsistent_conversion_sentinel_is_rejected(self):
        self.load([(0, 1, 1, 0, -1, 95, 0)])
        with self.assertRaisesRegex(ValueError, "invalid values"):
            source_summary(self.connection)

    def test_out_of_range_lookback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "from 1 to 30 days"):
            analyze(self.connection, 0)
