"""Leakage, comparison, and campaign-capacity checks using artificial records."""

import json
import unittest
from itertools import chain
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.model import (
    calibration_bins,
    choose_capacity_cutoff,
    chronological_slices,
    compare_call_policies,
    make_models,
    run_modeling,
)


def modeling_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """Provide both classes in each split and a category seen only after training."""
    positions = np.arange(40)
    outcomes = positions % 3 == 0
    contacted = positions % 2 == 1
    job = np.where(positions >= 24, "future-category", np.where(outcomes, "admin.", "technician"))
    features = pd.DataFrame({
        "balance": positions + np.where(outcomes, 100, 0),
        "previous": np.where(contacted, 2, 0),
        "contacted_before": contacted.astype(int),
        "prior_days": np.where(contacted, positions * 3, np.nan),
        "job": job,
        "education": ["secondary"] * 40,
        "default": ["no"] * 40,
        "housing": np.where(outcomes, "no", "yes"),
        "loan": ["no"] * 40,
        "poutcome": np.where(contacted, "failure", "unknown"),
    })
    return features, pd.Series(outcomes.astype(int), name="y")


class SplitTests(unittest.TestCase):
    def test_chronological_slices_are_contiguous_and_cover_the_source(self):
        slices = chronological_slices(11)
        parts = [list(range(span.start, span.stop)) for span in slices.values()]
        self.assertEqual([len(part) for part in parts], [6, 2, 3])
        self.assertEqual(list(chain.from_iterable(parts)), list(range(11)))

    def test_split_rejects_too_few_rows(self):
        with self.assertRaisesRegex(ValueError, "five ordered rows"):
            chronological_slices(4)


class ModelTests(unittest.TestCase):
    def test_rejects_unexpected_predictors_and_misaligned_targets(self):
        X, y = modeling_fixture()
        with self.assertRaisesRegex(ValueError, "pre-contact feature allowlist"):
            run_modeling(X.assign(duration=200), y)
        with self.assertRaisesRegex(ValueError, "Target must align"):
            run_modeling(X, y.iloc[::-1])

    def test_preprocessors_fit_only_on_training_and_metrics_cover_each_split(self):
        X, y = modeling_fixture()
        models = make_models(seed=42)
        with patch("src.model.make_models", return_value=models):
            metrics, scores = run_modeling(X, y)

        self.assertEqual(list(metrics["models"]), list(models))
        self.assertEqual([part["n"] for part in metrics["split"].values()], [24, 8, 8])
        for key, (_, model) in models.items():
            preprocessor = model.named_steps["preprocessor"]
            imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
            self.assertAlmostEqual(imputer.statistics_[0], X["balance"].iloc[:24].median())
            encoder = preprocessor.named_transformers_["categorical"]
            self.assertNotIn("future-category", encoder.categories_[0])
            for part in ("validation", "test"):
                result = metrics["models"][key][part]
                labels = scores[part]["y_true"]
                probabilities = scores[part]["probabilities"][key]
                self.assertEqual(probabilities.shape, labels.shape)
                self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
                self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (2, 2))
                self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), len(labels))
                self.assertEqual(sum(item["count"] for item in result["calibration"]), len(labels))

        selected = metrics["models"][metrics["selection"]["model"]]["validation"]["average_precision"]
        self.assertEqual(selected, max(model["validation"]["average_precision"] for model in metrics["models"].values()))
        self.assertLessEqual(metrics["policy"]["validation"]["fixed_threshold"]["call_fraction"], 0.20)
        self.assertAlmostEqual(metrics["models"]["dummy"]["test"]["average_precision"], y.iloc[32:].mean())
        json.dumps(metrics, allow_nan=False)


class CapacityTests(unittest.TestCase):
    def test_frozen_cutoff_excludes_the_boundary_tie_and_top_k_keeps_capacity(self):
        scores = np.array([0.9, 0.8, 0.8, 0.5, 0.4, 0.35, 0.3, 0.1, 0.1, 0.0])
        outcomes = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        cutoff = choose_capacity_cutoff(scores, 0.20)
        result = compare_call_policies(outcomes, scores, cutoff)

        self.assertEqual(cutoff, 0.8)
        self.assertEqual(result["fixed_threshold"]["calls"], 1)
        self.assertEqual(result["fixed_threshold"]["threshold"], cutoff)
        top_20 = result["capacity_scenarios"][1]
        self.assertEqual(top_20["calls"], 2)
        self.assertEqual(top_20["conversions"], 1)
        self.assertAlmostEqual(top_20["expected_random_conversions"], 0.6)

    def test_calibration_surfaces_empty_bins_and_includes_the_endpoint(self):
        bins = calibration_bins(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[-1]["mean_probability"], 1.0)
        self.assertEqual(bins[-1]["observed_rate"], 1.0)
        self.assertEqual(bins[4]["count"], 0)
        self.assertIsNone(bins[4]["observed_rate"])
