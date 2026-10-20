"""
5_evaluator.py
Evaluation helpers: ROC-AUC, threshold analysis, calibration and diagnostics.

The leaderboard metric is ROC-AUC, so every headline number reported here is
ROC-AUC. Accuracy / F1 are reported for the 0.5 operating point that the issue
specifies (TARGET >= 0.5 -> "jailbreak", otherwise "benign") and are secondary.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Compute the full metric suite for one score vector.

    Returns a dict with ROC-AUC, average precision, and the 0.5-threshold
    classification metrics required by the issue.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "accuracy@0.5": float(accuracy_score(y_true, y_pred)),
        "precision@0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    return metrics


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    """Render one metric dict as a single readable line."""
    return (
        f"  {name:<14} ROC-AUC={metrics['roc_auc']:.5f}  "
        f"AP={metrics['average_precision']:.5f}  "
        f"acc@0.5={metrics['accuracy@0.5']:.4f}  "
        f"F1@0.5={metrics['f1@0.5']:.4f}  "
        f"rec={metrics['recall@0.5']:.4f}  "
        f"prec={metrics['precision@0.5']:.4f}"
    )


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Confusion matrix at the given operating threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, float]]:
    """
    Sweep the decision threshold and report the accuracy-optimal point.

    ROC-AUC is threshold independent, but the exported TARGET column is
    binarised at 0.5 by the issue, so the sweep documents how much accuracy
    would change if a different cut-off were used.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows: List[Dict[str, float]] = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_score >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        })
    return rows


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold maximising F1 (reported for information only)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(threshold)
    return best_t


def score_distribution(y_score: np.ndarray) -> Dict[str, float]:
    """Summary statistics of the exported score distribution."""
    y_score = np.asarray(y_score, dtype=float)
    return {
        "min": float(y_score.min()),
        "p01": float(np.quantile(y_score, 0.01)),
        "p05": float(np.quantile(y_score, 0.05)),
        "p25": float(np.quantile(y_score, 0.25)),
        "median": float(np.median(y_score)),
        "p75": float(np.quantile(y_score, 0.75)),
        "p95": float(np.quantile(y_score, 0.95)),
        "p99": float(np.quantile(y_score, 0.99)),
        "max": float(y_score.max()),
        "mean": float(y_score.mean()),
        "std": float(y_score.std()),
        "frac_below_0.5": float((y_score < 0.5).mean()),
        "frac_above_0.5": float((y_score >= 0.5).mean()),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval for the ROC-AUC.

    A confidence interval is what separates "this model is genuinely better"
    from "this model got a lucky split", so it is reported alongside the point
    estimate for every candidate.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    scores: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    lower = float(np.quantile(scores, alpha / 2))
    upper = float(np.quantile(scores, 1 - alpha / 2))
    return (lower, upper)


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 11) -> np.ndarray:
    """Sample points of the ROC curve for plotting/inspection."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    indices = np.linspace(0, len(fpr) - 1, n_points).astype(int)
    return np.column_stack([fpr[indices], tpr[indices]])
