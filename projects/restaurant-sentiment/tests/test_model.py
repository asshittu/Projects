"""Check split isolation, train-only vocabulary, and error excerpt selection."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.model import error_candidates, make_models, run_modeling, stratified_splits


def fixture() -> pd.DataFrame:
    """Build short synthetic reviews with one unique token per record."""
    text = [
        f"{'Delicious friendly wonderful' if index % 2 else 'Awful rude terrible'} signatureword{index:03d}"
        for index in range(40)
    ]
    return pd.DataFrame({"text": text, "label": np.arange(40) % 2}).astype({"text": "string"})


class SplitTests(unittest.TestCase):
    def test_seeded_stratified_splits_are_disjoint_and_cover_the_data(self):
        data = fixture()
        splits = stratified_splits(data["label"], seed=42)
        same_seed = stratified_splits(data["label"], seed=42)
        self.assertEqual([len(part) for part in splits.values()], [24, 8, 8])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(data)))
        for part, positions in splits.items():
            np.testing.assert_array_equal(positions, same_seed[part])
            self.assertEqual(data["label"].iloc[positions].mean(), 0.50)

    def test_rejects_equal_review_keys_before_any_split(self):
        data = fixture()
        data.loc[1, "text"] = "  ＡＷＦＵＬ \t rude  terrible  SIGNATUREWORD000 "
        with self.assertRaisesRegex(ValueError, "deduplicated before splitting"):
            run_modeling(data)


class ModelTests(unittest.TestCase):
    def test_baseline_vocabulary_and_holdout_metrics(self):
        data = fixture()
        models = make_models()
        with patch("src.model.make_models", return_value=models):
            metrics, details = run_modeling(data)

        self.assertEqual(metrics["selection"]["model"], "tfidf_unigram")
        for key in ("tfidf_unigram", "tfidf_unigram_bigram"):
            vocabulary = models[key][1].named_steps["tfidf"].vocabulary_
            for position in details["train"]["positions"]:
                self.assertIn(f"signatureword{position:03d}", vocabulary)
            for part in ("validation", "test"):
                for position in details[part]["positions"]:
                    self.assertNotIn(f"signatureword{position:03d}", vocabulary)

        test = details["test"]
        dummy = metrics["models"]["dummy"]["test"]
        np.testing.assert_allclose(test["probabilities"]["dummy"], 0.50)
        np.testing.assert_array_equal(test["predictions"]["dummy"], np.zeros(8))
        self.assertEqual(dummy["confusion_matrix"], [[4, 0], [4, 0]])
        self.assertEqual(dummy["class_metrics"]["negative"]["recall"], 1.0)
        self.assertEqual(dummy["class_metrics"]["positive"]["precision"], 0.0)
        self.assertAlmostEqual(dummy["macro_f1"], 1 / 3)
        self.assertAlmostEqual(dummy["average_precision"], 0.50)
        json.dumps(metrics, allow_nan=False)

    def test_error_examples_are_bounded_and_ordered_by_score(self):
        reviews = pd.Series(["sample"] * 8)
        reviews.iloc[2] = "A long synthetic review. " * 20
        outcomes = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        predicted = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        probabilities = np.array([0.6, 0.8, 0.9, 0.4, 0.1, 0.2, 0.3, 0.8])
        errors = error_candidates(reviews, np.arange(8), outcomes, predicted, probabilities)
        self.assertEqual([item["position"] for item in errors["false_positive"]], [2, 1])
        self.assertEqual([item["position"] for item in errors["false_negative"]], [4, 5])
        self.assertLessEqual(len(errors["false_positive"][0]["excerpt"]), 140)
        self.assertTrue(errors["false_positive"][0]["excerpt"].endswith("…"))
