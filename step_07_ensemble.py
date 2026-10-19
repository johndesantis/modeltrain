"""Rank fusion, calibration, and full-data prediction (step 7).

The six base models emit scores on very different scales (probabilities for
logistic regression / Naive Bayes / KNN / LightGBM, margins for the linear
SVM).  ROC-AUC only cares about ordering, so each model's scores are converted
to within-set percentile ranks and averaged with fixed weights.  This is the
same procedure used during cross-validation and at submission time.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from step_01_config import CALIBRATION_EPS, MODEL_NAMES
from step_05_models import fit_and_score


def rank_normalize(scores: Sequence[float]) -> np.ndarray:
    """Map scores to ``(0, 1)`` percentile ranks (ties get average ranks)."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("scores contain NaN or infinite values")
    return rankdata(values, method="average") / (values.size + 1.0)


def blend_ranked(
    score_map: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of per-model percentile ranks."""
    if not score_map:
        raise ValueError("score_map is empty")
    if weights is None:
        weights = {name: 1.0 for name in score_map}
    missing = set(score_map) - set(weights)
    if missing:
        raise KeyError(f"No blend weight supplied for: {sorted(missing)}")
    total_weight = float(sum(weights[name] for name in score_map))
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value")

    lengths = {len(np.asarray(scores).ravel()) for scores in score_map.values()}
    if len(lengths) != 1:
        raise ValueError(f"All score arrays must have the same length, got {lengths}")

    blended = np.zeros(next(iter(lengths)), dtype=np.float64)
    for name, scores in score_map.items():
        blended += float(weights[name]) * rank_normalize(scores)
    return blended / total_weight


def fit_calibrator(oof_scores: Sequence[float], labels: Sequence[int]) -> LogisticRegression:
    """Fit Platt scaling (one-feature logistic regression) on OOF scores."""
    scores = np.asarray(oof_scores, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("oof_scores and labels must have the same length")
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    calibrator.fit(scores, labels)
    return calibrator


def apply_calibrator(calibrator: LogisticRegression, scores: Sequence[float]) -> np.ndarray:
    """Apply the calibrator and clip to ``[eps, 1 - eps]``."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    probabilities = calibrator.predict_proba(values)[:, 1]
    return np.clip(probabilities, CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)


def fit_full_models(
    X_train,
    y_train: np.ndarray,
    X_test,
    safeguard_train: np.ndarray,
    safeguard_test: np.ndarray,
    model_names: Sequence[str] = MODEL_NAMES,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Fit every base model on all training rows and score the test rows."""
    predictions: dict[str, np.ndarray] = {}
    for name in model_names:
        if verbose:
            print(f"fitting {name} on all {X_train.shape[0]} training rows ...", flush=True)
        predictions[name] = fit_and_score(
            name,
            X_train,
            y_train,
            X_test,
            safeguard_train=safeguard_train,
            safeguard_score=safeguard_test,
        )
        if verbose:
            print(f"  {name}: scored {len(predictions[name])} test rows", flush=True)
    return predictions


def prediction_summary(
    scores: Sequence[float], threshold: float = 0.5
) -> dict[str, object]:
    """Descriptive statistics for an exported score column."""
    values = np.asarray(scores, dtype=np.float64)
    return {
        "rows": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_at_or_above_0.5": float((values >= threshold).mean()),
        "fraction_below_0.5": float((values < threshold).mean()),
        "unique_values": int(np.unique(np.round(values, 6)).size),
    }


def correlation_between_models(score_map: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Spearman correlation of each model with the blended rank (diagnostic)."""
    blended = blend_ranked(score_map)
    correlations = {}
    for name, scores in score_map.items():
        ranks = rank_normalize(scores)
        correlations[name] = float(np.corrcoef(ranks, blended)[0, 1])
    return correlations
