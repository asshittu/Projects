"""Artificial-fixture checks for split isolation, training, and outreach rules."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.model import (
    calibration_bins,
    choose_capacity_cutoff,
    compare_outreach_policies,
    make_models,
    run_modeling,
    stratified_profile_split,
)


def modeling_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """Create forty profiles, each repeated twice, including undefined call averages."""
    profile = np.tile(np.arange(40), 2)
    zero_calls = profile % 5 == 0
    calls = np.where(zero_calls, 0, profile + 10)
    seconds = calls * (profile + 20)
    average_call_seconds = np.divide(
        seconds, calls, out=np.full(len(profile), np.nan), where=calls != 0,
    )
    index = np.arange(100, 100 + 3 * len(profile), 3)
    X = pd.DataFrame({
        "call_failure": profile // 3,
        "complains": (profile % 3 == 0).astype(int),
        "subscription_length": 12 + profile,
        "charge_amount": profile % 11,
        "seconds_of_use": seconds,
        "frequency_of_use": calls,
        "frequency_of_sms": profile ** 2,
        "distinct_called_numbers": np.maximum(0, calls - 6),
        "average_call_seconds": average_call_seconds,
        "no_voice_calls": zero_calls.astype(int),
        "tariff_plan": 1 + profile % 2,
    }, index=index)
    y = pd.Series((profile % 4 == 0).astype(int), index=index, name="churn")
    # One identical profile has conflicting outcomes and must still stay together.
    y.iloc[41] = 1
    return X, y


class ProfileSplitTests(unittest.TestCase):
    def test_matches_seeded_folds_and_prevents_overlap_even_with_missing_values(self):
        X, y = modeling_fixture()
        splits, groups = stratified_profile_split(X, y, seed=42)
        expected_groups = X.groupby(list(X.columns), dropna=False).ngroup().to_numpy()
        np.testing.assert_array_equal(groups, expected_groups)

        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = [positions for _, positions in splitter.split(X, y, expected_groups)]
        np.testing.assert_array_equal(splits["test"], np.sort(folds[0]))
        np.testing.assert_array_equal(splits["validation"], np.sort(folds[1]))
        np.testing.assert_array_equal(splits["train"], np.sort(np.concatenate(folds[2:])))
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(X)))

        partition_groups = [set(groups[positions]) for positions in splits.values()]
        self.assertEqual(sum(map(len, partition_groups)), len(set().union(*partition_groups)))
        self.assertEqual(groups[0], groups[40])
        self.assertEqual(groups[1], groups[41])
        for positions in splits.values():
            self.assertEqual(set(y.iloc[positions]), {0, 1})

    def test_requires_five_distinct_profiles(self):
        X, y = modeling_fixture()
        positions = [0, 1, 2, 3, 40, 41, 42, 43]
        with self.assertRaisesRegex(ValueError, "five distinct predictor profiles"):
            stratified_profile_split(X.iloc[positions], y.iloc[positions])


class ModelTests(unittest.TestCase):
    def test_rejects_outcome_like_predictors_and_misaligned_target(self):
        X, y = modeling_fixture()
        with self.assertRaisesRegex(ValueError, "pre-outreach feature allowlist"):
            run_modeling(X.assign(status=1), y)
        with self.assertRaisesRegex(ValueError, "Target must align"):
            run_modeling(X, y.iloc[::-1])

    def test_preprocessing_comes_from_training_and_metrics_cover_holdout(self):
        X, y = modeling_fixture()
        models = make_models(seed=42)
        with patch("src.model.make_models", return_value=models):
            metrics, scores = run_modeling(X, y, seed=42)

        train = X.iloc[scores["train"]["positions"]]
        self.assertEqual(list(metrics["models"]), list(models))
        self.assertEqual(sum(part["n"] for part in metrics["split"].values()), len(X))
        self.assertEqual(sum(part["profile_groups"] for part in metrics["split"].values()), 40)
        for key, (_, model) in models.items():
            preprocessor = model.named_steps["preprocessor"]
            imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
            np.testing.assert_allclose(
                imputer.statistics_, train[preprocessor.transformers_[0][2]].median().to_numpy(),
            )
            for part in ("validation", "test"):
                result = metrics["models"][key][part]
                labels = scores[part]["y_true"]
                probabilities = scores[part]["probabilities"][key]
                np.testing.assert_array_equal(labels, y.iloc[scores[part]["positions"]].to_numpy())
                self.assertEqual(probabilities.shape, labels.shape)
                self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
                self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (2, 2))
                self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), len(labels))
                self.assertEqual(sum(item["count"] for item in result["calibration"]), len(labels))

        chosen = metrics["models"][metrics["selection"]["model"]]["validation"]["average_precision"]
        self.assertEqual(chosen, max(model["validation"]["average_precision"] for model in metrics["models"].values()))
        self.assertLessEqual(metrics["policy"]["validation"]["fixed_threshold"]["call_fraction"], 0.20)
        self.assertAlmostEqual(metrics["models"]["dummy"]["test"]["average_precision"], scores["test"]["y_true"].mean())
        json.dumps(metrics, allow_nan=False)


class CapacityTests(unittest.TestCase):
    def test_frozen_cutoff_respects_tie_and_ranked_rule_fills_budget(self):
        scores = np.array([0.9, 0.8, 0.8, 0.5, 0.4, 0.35, 0.3, 0.1, 0.1, 0.0])
        outcomes = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        cutoff = choose_capacity_cutoff(scores, 0.20)
        result = compare_outreach_policies(outcomes, scores, cutoff)

        self.assertEqual(cutoff, 0.8)
        self.assertEqual(result["fixed_threshold"]["calls"], 1)
        self.assertEqual(result["fixed_threshold"]["threshold"], cutoff)
        top_20 = result["capacity_scenarios"][1]
        self.assertEqual(top_20["calls"], 2)
        self.assertEqual(top_20["churners"], 1)
        self.assertAlmostEqual(top_20["expected_random_churners"], 0.6)

    def test_calibration_includes_empty_bins_and_upper_endpoint(self):
        bins = calibration_bins(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[-1]["mean_probability"], 1.0)
        self.assertEqual(bins[-1]["observed_rate"], 1.0)
        self.assertEqual(bins[4]["count"], 0)
        self.assertIsNone(bins[4]["observed_rate"])
