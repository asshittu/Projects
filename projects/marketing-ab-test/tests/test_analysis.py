"""Artificial-fixture tests for the primary conversion comparison."""

import json
import unittest

import pandas as pd
from src.analysis import run_analysis


def fixture(n_ad: int, converted_ad: int, n_psa: int, converted_psa: int) -> pd.DataFrame:
    """Construct one row and a Boolean outcome for each test user."""
    return pd.DataFrame({
        "test_group": ["ad"] * n_ad + ["psa"] * n_psa,
        "converted": (
            [True] * converted_ad + [False] * (n_ad - converted_ad)
            + [True] * converted_psa + [False] * (n_psa - converted_psa)
        ),
    })


class AnalysisTests(unittest.TestCase):
    def test_unequal_arms_have_correct_rates_intervals_and_contrast(self):
        result = run_analysis(fixture(100, 25, 50, 5))
        ad = result["groups"]["ad"]
        psa = result["groups"]["psa"]
        contrast = result["contrast"]

        self.assertEqual((ad["n"], ad["converted"], ad["conversion_rate"]), (100, 25, 0.25))
        self.assertEqual((psa["n"], psa["converted"], psa["conversion_rate"]), (50, 5, 0.10))
        self.assertAlmostEqual(ad["observed_share"], 2 / 3)
        self.assertAlmostEqual(psa["observed_share"], 1 / 3)
        self.assertAlmostEqual(ad["wilson_95_ci"]["lower"], 0.17545211362287674)
        self.assertAlmostEqual(psa["wilson_95_ci"]["upper"], 0.21360231437479654)
        self.assertEqual(contrast["absolute_difference"], 0.15)
        self.assertAlmostEqual(contrast["relative_lift"], 1.50)
        self.assertAlmostEqual(contrast["newcombe_95_ci"]["lower"], 0.01412184430672242)
        self.assertAlmostEqual(contrast["newcombe_95_ci"]["upper"], 0.25886823844333856)
        self.assertAlmostEqual(contrast["pooled_z_statistic"], 2.1650635094610964)
        self.assertAlmostEqual(contrast["two_sided_p_value"], 0.03038282197657753)
        self.assertAlmostEqual(contrast["additional_conversions_per_10000"], 1500)
        self.assertAlmostEqual(contrast["additional_conversions_95_ci"]["lower"], 141.2184430672242)
        self.assertTrue(result["practical_hurdle"]["point_estimate_meets_hurdle"])
        self.assertEqual(result["practical_hurdle"]["interval_relation"], "entirely_at_or_above")
        self.assertFalse(result["design"]["planned_allocation_known"])
        self.assertFalse(result["design"]["sample_ratio_mismatch_tested"])
        json.dumps(result, allow_nan=False)

    def test_post_exposure_columns_have_no_effect_on_primary_estimation(self):
        data = fixture(20, 4, 5, 1)
        original = run_analysis(data)
        post_exposure = data.assign(total_ads=1_000, most_ads_day="Monday", most_ads_hour=11)
        post_exposure["total_ads"] = post_exposure["converted"].map({True: 1000, False: 0})
        self.assertEqual(run_analysis(post_exposure), original)
        self.assertEqual(original["contrast"]["two_sided_p_value"], 1.0)
        self.assertFalse(original["practical_hurdle"]["point_estimate_meets_hurdle"])
        self.assertEqual(original["practical_hurdle"]["interval_relation"], "spans_hurdle")

    def test_zero_comparator_rate_and_degenerate_z_are_serializable(self):
        result = run_analysis(fixture(20, 1, 10, 0))
        self.assertIsNone(result["contrast"]["relative_lift"])
        self.assertIsNotNone(result["contrast"]["two_sided_p_value"])

        for data in (fixture(20, 0, 10, 0), fixture(20, 20, 10, 10)):
            extreme = run_analysis(data)
            self.assertEqual(extreme["contrast"]["absolute_difference"], 0.0)
            self.assertIsNone(extreme["contrast"]["pooled_z_statistic"])
            self.assertIsNone(extreme["contrast"]["two_sided_p_value"])
            json.dumps(extreme, allow_nan=False)

    def test_negative_contrast_and_interval_are_kept(self):
        result = run_analysis(fixture(100, 5, 100, 30))
        self.assertAlmostEqual(result["contrast"]["absolute_difference"], -0.25)
        self.assertLess(result["contrast"]["newcombe_95_ci"]["upper"], 0.0)
        self.assertEqual(result["practical_hurdle"]["interval_relation"], "entirely_below")

    def test_rejects_missing_columns_groups_or_boolean_outcomes(self):
        with self.assertRaisesRegex(ValueError, "test_group and converted"):
            run_analysis(pd.DataFrame({"converted": [False, True]}))
        with self.assertRaisesRegex(ValueError, "nonempty ad and psa"):
            run_analysis(pd.DataFrame({"test_group": ["ad", "other"], "converted": [False, True]}))
        with self.assertRaisesRegex(ValueError, "nonempty ad and psa"):
            run_analysis(pd.DataFrame({"test_group": ["ad"], "converted": [False]}))
        with self.assertRaisesRegex(ValueError, "nonmissing Booleans"):
            run_analysis(pd.DataFrame({"test_group": ["ad", "psa"], "converted": [1, 0]}))
        with self.assertRaisesRegex(ValueError, "nonmissing Booleans"):
            run_analysis(pd.DataFrame({
                "test_group": ["ad", "psa"],
                "converted": pd.array([False, pd.NA], dtype="boolean"),
            }))
