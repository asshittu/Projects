"""Chronological model comparison and validation-set call policy selection."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

POLICY_CAPACITY = 0.20
CAPACITY_SCENARIOS = (0.10, 0.20, 0.30)


def chronological_slices(n_rows: int) -> dict[str, slice]:
    """Partition the publisher's ordered rows into consecutive 60/20/20 blocks."""
    if n_rows < 5:
        raise ValueError("At least five ordered rows are required for three splits")
    validation_start = 3 * n_rows // 5
    test_start = 4 * n_rows // 5
    return {
        "train": slice(0, validation_start),
        "validation": slice(validation_start, test_start),
        "test": slice(test_start, n_rows),
    }


def make_preprocessor() -> ColumnTransformer:
    """Normalize numeric predictors and encode categories within each training pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])


def make_models(seed: int) -> dict[str, tuple[str, Pipeline]]:
    """Create fixed candidates sharing the same feature availability rules."""
    estimators = {
        "dummy": ("Dummy (training prevalence)", DummyClassifier(strategy="prior")),
        "logistic_regression": ("Logistic regression", LogisticRegression(max_iter=1000)),
        "random_forest": ("Random forest", RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=30,
            n_jobs=1,
            random_state=seed,
        )),
    }
    return {
        key: (label, Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", estimator),
        ]))
        for key, (label, estimator) in estimators.items()
    }


def calibration_bins(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Summarize reliability using fixed-width bins, retaining the empty bins."""
    indices = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    bins = []
    for index in range(n_bins):
        members = indices == index
        count = int(members.sum())
        bins.append({
            "lower": index / n_bins,
            "upper": (index + 1) / n_bins,
            "mean_probability": float(scores[members].mean()) if count else None,
            "observed_rate": float(y_true[members].mean()) if count else None,
            "count": count,
        })
    return bins


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compare probability quality and classification at a common 0.50 cutoff."""
    predicted = scores >= 0.50
    return {
        "threshold": 0.50,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "calibration": calibration_bins(y_true, scores),
    }


def choose_capacity_cutoff(scores: np.ndarray, capacity: float) -> float:
    """Return the first score outside the budget for a strict greater-than rule."""
    if len(scores) == 0 or not 0 <= capacity < 1:
        raise ValueError("A nonempty score array and capacity in [0, 1) are required")
    budget = int(len(scores) * capacity)
    return float(np.sort(scores)[::-1][budget])


def policy_summary(y_true: np.ndarray, called: np.ndarray) -> dict:
    """Measure observed outcome capture and compare to uniform random selection."""
    actual_positives = int(y_true.sum())
    prevalence = actual_positives / len(y_true)
    calls = int(called.sum())
    conversions = int(y_true[called].sum())
    precision = conversions / calls if calls else None
    return {
        "calls": calls,
        "call_fraction": calls / len(y_true),
        "conversions": conversions,
        "precision": precision,
        "recall": conversions / actual_positives,
        "f1": 2 * conversions / (actual_positives + calls),
        "confusion_matrix": confusion_matrix(y_true, called, labels=[0, 1]).tolist(),
        "lift": precision / prevalence if precision is not None else None,
        "expected_random_conversions": calls * prevalence,
    }


def compare_call_policies(y_true: np.ndarray, scores: np.ndarray, cutoff: float) -> dict:
    """Evaluate a frozen cutoff and batch-ranking rules at specified call volumes."""
    fixed = policy_summary(y_true, scores > cutoff)
    fixed["threshold"] = cutoff

    # Equal scores follow the source row order for reproducible retrospective rankings.
    descending = np.argsort(-scores, kind="stable")
    scenarios = []
    for capacity in CAPACITY_SCENARIOS:
        calls = int(len(scores) * capacity)
        called = np.zeros(len(scores), dtype=bool)
        called[descending[:calls]] = True
        scenario = policy_summary(y_true, called)
        scenario["capacity"] = capacity
        scenario["threshold"] = float(scores[descending[calls - 1]]) if calls else None
        scenarios.append(scenario)
    return {"fixed_threshold": fixed, "capacity_scenarios": scenarios}


def run_modeling(X: pd.DataFrame, y: pd.Series, *, seed: int = 42) -> tuple[dict, dict]:
    """Fit on train, select from validation, then evaluate the frozen rules on test."""
    expected_features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    if not X.columns.is_unique or set(X.columns) != expected_features:
        raise ValueError("Predictors must match the pre-contact feature allowlist")
    if not X.index.equals(y.index) or not y.isin([0, 1]).all():
        raise ValueError("Target must align with predictors and contain only 0 or 1")

    splits = chronological_slices(len(X))
    outcomes = {name: y.iloc[span].to_numpy(dtype=np.int8) for name, span in splits.items()}
    if any(np.unique(values).size != 2 for values in outcomes.values()):
        raise ValueError("Each chronological split must contain both outcome classes")
    train_X = X.iloc[splits["train"]]
    validation_X = X.iloc[splits["validation"]]
    test_X = X.iloc[splits["test"]]

    models = make_models(seed)
    scores = {
        part: {"y_true": outcomes[part], "probabilities": {}}
        for part in ("validation", "test")
    }
    comparisons = {}
    for key, (label, model) in models.items():
        model.fit(train_X, outcomes["train"])
        probabilities = model.predict_proba(validation_X)[:, 1]
        scores["validation"]["probabilities"][key] = probabilities
        comparisons[key] = {"label": label, "validation": evaluate(outcomes["validation"], probabilities)}

    selected_key = max(comparisons, key=lambda key: comparisons[key]["validation"]["average_precision"])
    selected_validation_scores = scores["validation"]["probabilities"][selected_key]
    cutoff = choose_capacity_cutoff(selected_validation_scores, POLICY_CAPACITY)

    for key, (_, model) in models.items():
        probabilities = model.predict_proba(test_X)[:, 1]
        scores["test"]["probabilities"][key] = probabilities
        comparisons[key]["test"] = evaluate(outcomes["test"], probabilities)

    metrics = {
        "seed": seed,
        "features": {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES)},
        "split": {
            part: {
                "start": span.start,
                "stop": span.stop,
                "n": len(outcomes[part]),
                "positives": int(outcomes[part].sum()),
                "prevalence": float(outcomes[part].mean()),
            }
            for part, span in splits.items()
        },
        "selection": {
            "model": selected_key,
            "criterion": "validation_average_precision",
            "threshold": cutoff,
            "comparison": ">",
            "validation_capacity": POLICY_CAPACITY,
            "threshold_method": "First validation score outside the ranked call budget, using a strict greater-than comparison",
        },
        "models": comparisons,
        "policy": {
            part: compare_call_policies(outcomes[part], scores[part]["probabilities"][selected_key], cutoff)
            for part in ("validation", "test")
        },
    }
    return metrics, scores
