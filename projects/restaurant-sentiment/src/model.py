"""Compare text classifiers with vocabulary fitted only to the training reviews."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import text_keys

CLASS_LABELS = ("negative", "positive")
MAX_ERROR_EXCERPTS = 2
MAX_EXCERPT_LENGTH = 140


def stratified_splits(y: pd.Series, *, seed: int = 42) -> dict[str, np.ndarray]:
    """Return disjoint positional indices near 60/20/20, stratified by label."""
    positions = np.arange(len(y))
    train, remaining = train_test_split(positions, test_size=0.40, stratify=y, random_state=seed)
    validation, test = train_test_split(
        remaining, test_size=0.50, stratify=y.iloc[remaining], random_state=seed,
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def make_models() -> dict[str, tuple[str, DummyClassifier | Pipeline]]:
    """Create candidates in simplest-first order for validation-score ties."""
    models = {"dummy": ("Dummy (training prior)", DummyClassifier(strategy="prior"))}
    for key, label, ngram_range in (
        ("tfidf_unigram", "TF-IDF unigram + logistic regression", (1, 1)),
        ("tfidf_unigram_bigram", "TF-IDF unigram and bigram + logistic regression", (1, 2)),
    ):
        models[key] = (label, Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]))
    return models


def evaluate(y_true: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    """Report forced binary predictions, class metrics, and positive-label ranking."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1], zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(np.mean(f1)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "class_metrics": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
    }


def error_candidates(
    text: pd.Series, positions: np.ndarray, true: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray,
) -> dict[str, list[dict]]:
    """Return the highest-scored false positives and lowest-scored false negatives for review."""
    candidates = {}
    for key, condition, sort_scores in (
        ("false_positive", (true == 0) & (predicted == 1), -probabilities),
        ("false_negative", (true == 1) & (predicted == 0), probabilities),
    ):
        eligible = np.flatnonzero(condition)
        order = np.argsort(sort_scores[eligible], kind="stable")
        records = []
        for index in eligible[order[:MAX_ERROR_EXCERPTS]]:
            position = int(positions[index])
            excerpt = " ".join(text.iloc[position].split())
            if len(excerpt) > MAX_EXCERPT_LENGTH:
                excerpt = excerpt[:MAX_EXCERPT_LENGTH - 1].rstrip() + "…"
            records.append({
                "position": position,
                "excerpt": excerpt,
                "positive_probability": float(probabilities[index]),
                "true_label": int(true[index]),
                "predicted_label": int(predicted[index]),
            })
        candidates[key] = records
    return candidates


def run_modeling(data: pd.DataFrame, *, seed: int = 42) -> tuple[dict, dict]:
    """Fit candidates on train, select by validation macro F1, and audit the holdout."""
    if not {"text", "label"} <= set(data.columns):
        raise ValueError("Modeling requires text and label columns")
    text, y = data["text"], data["label"]
    if text.isna().any():
        raise ValueError("Review text must be nonmissing")
    keys = text_keys(text)
    if keys.eq("").any() or keys.duplicated().any():
        raise ValueError("Review text must be nonempty and deduplicated before splitting")
    if not y.isin([0, 1]).all() or y.value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError("Modeling requires at least five reviews for each binary label")

    splits = stratified_splits(y, seed=seed)
    scores = {
        part: {"positions": positions, "y_true": y.iloc[positions].to_numpy(dtype=np.int8)}
        for part, positions in splits.items()
    }
    for part in ("validation", "test"):
        scores[part]["probabilities"] = {}
        scores[part]["predictions"] = {}

    models = make_models()
    results = {}
    training_reviews = text.iloc[splits["train"]]
    validation_reviews = text.iloc[splits["validation"]]
    test_reviews = text.iloc[splits["test"]]
    for key, (label, model) in models.items():
        model.fit(training_reviews, scores["train"]["y_true"])
        predictions = model.predict(validation_reviews)
        probabilities = model.predict_proba(validation_reviews)[:, 1]
        scores["validation"]["predictions"][key] = predictions
        scores["validation"]["probabilities"][key] = probabilities
        results[key] = {"label": label, "validation": evaluate(scores["validation"]["y_true"], predictions, probabilities)}

    selected = max(results, key=lambda key: results[key]["validation"]["macro_f1"])
    for key, (_, model) in models.items():
        predictions = model.predict(test_reviews)
        probabilities = model.predict_proba(test_reviews)[:, 1]
        scores["test"]["predictions"][key] = predictions
        scores["test"]["probabilities"][key] = probabilities
        results[key]["test"] = evaluate(scores["test"]["y_true"], predictions, probabilities)

    metrics = {
        "seed": seed,
        "split_strategy": "Seeded stratified 60/20/20 positional split after normalized text deduplication",
        "split": {
            part: {
                "n": len(details["positions"]),
                "negative": int((details["y_true"] == 0).sum()),
                "positive": int(details["y_true"].sum()),
                "positive_rate": float(details["y_true"].mean()),
            }
            for part, details in scores.items()
        },
        "selection": {
            "model": selected,
            "criterion": "validation_macro_f1",
            "tie_breaker": "Fixed candidate order, simplest first",
        },
        "models": results,
    }
    test = scores["test"]
    scores["errors"] = error_candidates(
        text, test["positions"], test["y_true"], test["predictions"][selected], test["probabilities"][selected],
    )
    return metrics, scores
