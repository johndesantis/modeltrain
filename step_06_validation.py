"""Grouped cross-validation (step 6).

Random K-fold would let near-duplicate prompts appear in both the training and
validation side of a fold, inflating the score.  ``StratifiedGroupKFold`` keeps
every near-duplicate cluster in a single fold, so the reported ROC-AUC is a
more honest estimate for unseen prompts.

For every fold we fit the TF-IDF union once, then fit all six base models on
the same matrix.  The ensemble score for the fold is the weighted rank average
of the base models, exactly as used for the final submission.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from step_01_config import (
    BLEND_WEIGHTS,
    CV_SEEDS,
    MODEL_NAMES,
    N_SPLITS,
)
from step_04_features import build_baseline_vectorizer, build_tfidf_union
from step_05_models import fit_and_score, make_baseline_model
from step_07_ensemble import blend_ranked, rank_normalize


def grouped_folds(
    labels: Sequence[int],
    groups: Sequence[int],
    n_splits: int = N_SPLITS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified group folds as ``(train_idx, val_idx)``."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_idx), np.asarray(val_idx))
        for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels, groups)
    ]


def check_fold_integrity(
    folds: Iterable[tuple[np.ndarray, np.ndarray]], groups: Sequence[int]
) -> dict[str, object]:
    """Verify folds cover every row once and never split a duplicate group."""
    groups = np.asarray(groups)
    seen = np.zeros(len(groups), dtype=np.int64)
    leaked_groups = 0
    for train_idx, val_idx in folds:
        seen[val_idx] += 1
        overlap = set(groups[train_idx]).intersection(set(groups[val_idx]))
        leaked_groups += len(overlap)
    return {
        "every_row_validated_once": bool((seen == 1).all()),
        "groups_split_across_folds": leaked_groups,
    }


def _evaluate_fold(
    texts: Sequence[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    safeguard_matrix: np.ndarray,
    model_names: Sequence[str],
    weights: dict[str, float],
) -> tuple[dict[str, float], float, np.ndarray]:
    """Fit all models for one fold; return per-model AUCs and blend AUC."""
    vectorizer = build_tfidf_union()
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    scores: dict[str, np.ndarray] = {}
    model_aucs: dict[str, float] = {}
    for name in model_names:
        score = fit_and_score(
            name,
            X_train,
            y_train,
            X_val,
            safeguard_train=safeguard_matrix[train_idx],
            safeguard_score=safeguard_matrix[val_idx],
        )
        scores[name] = score
        model_aucs[name] = float(roc_auc_score(y_val, score))

    blended = blend_ranked(scores, weights)
    ensemble_auc = float(roc_auc_score(y_val, blended))
    return model_aucs, ensemble_auc, rank_normalize(blended)


def evaluate_cv(
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[int],
    safeguard_matrix: np.ndarray,
    seeds: Sequence[int] = CV_SEEDS,
    n_splits: int = N_SPLITS,
    model_names: Sequence[str] = MODEL_NAMES,
    weights: dict[str, float] | None = None,
    include_baseline: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Run grouped cross-validation for the baseline, all models, and the blend."""
    weights = dict(BLEND_WEIGHTS if weights is None else weights)
    labels = np.asarray(labels, dtype=np.int64)
    fold_aucs: dict[str, list[float]] = {name: [] for name in model_names}
    baseline_aucs: list[float] = []
    ensemble_fold_aucs: list[float] = []
    ensemble_per_seed: dict[str, dict[str, float]] = {}
    oof_rank_by_seed: dict[int, np.ndarray] = {}
    fold_metadata: list[dict[str, object]] = []
    integrity: dict[str, object] = {}

    for seed in seeds:
        folds = grouped_folds(labels, groups, n_splits=n_splits, seed=seed)
        integrity[str(seed)] = check_fold_integrity(folds, groups)
        oof_rank = np.zeros(len(labels), dtype=np.float64)
        seed_ensemble_aucs: list[float] = []

        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            model_aucs, blend_auc, ranked = _evaluate_fold(
                texts,
                labels,
                train_idx,
                val_idx,
                safeguard_matrix,
                model_names,
                weights,
            )
            oof_rank[val_idx] = ranked
            for name, value in model_aucs.items():
                fold_aucs[name].append(value)
            ensemble_fold_aucs.append(blend_auc)
            seed_ensemble_aucs.append(blend_auc)
            if verbose:
                print(
                    f"[seed {seed} | fold {fold_number}/{n_splits}] "
                    f"n_train={len(train_idx)} n_val={len(val_idx)} "
                    f"ensemble_auc={blend_auc:.4f}",
                    flush=True,
                )

            if include_baseline:
                baseline_vectorizer = build_baseline_vectorizer()
                baseline_train = baseline_vectorizer.fit_transform(
                    [texts[i] for i in train_idx]
                )
                baseline_val = baseline_vectorizer.transform([texts[i] for i in val_idx])
                baseline_model = make_baseline_model()
                baseline_model.fit(baseline_train, labels[train_idx])
                baseline_probability = baseline_model.predict_proba(baseline_val)[:, 1]
                baseline_aucs.append(float(roc_auc_score(labels[val_idx], baseline_probability)))

            fold_metadata.append(
                {
                    "seed": int(seed),
                    "fold": fold_number,
                    "n_train": int(len(train_idx)),
                    "n_validation": int(len(val_idx)),
                    "validation_positive_rate": float(labels[val_idx].mean()),
                }
            )

        oof_rank_by_seed[int(seed)] = oof_rank
        ensemble_per_seed[str(seed)] = {
            "mean_fold_auc": float(np.mean(seed_ensemble_aucs)),
            "std_fold_auc": float(np.std(seed_ensemble_aucs)),
        }

    per_model = {}
    for name in model_names:
        values = np.asarray(fold_aucs[name], dtype=np.float64)
        per_model[name] = {
            "mean_auc": float(values.mean()),
            "std_auc": float(values.std()),
            "fold_aucs": values.round(6).tolist(),
        }

    baseline_values = np.asarray(baseline_aucs, dtype=np.float64)
    ensemble_values = np.asarray(ensemble_fold_aucs, dtype=np.float64)
    return {
        "seeds": [int(seed) for seed in seeds],
        "n_splits": int(n_splits),
        "model_names": list(model_names),
        "weights": {name: float(weights[name]) for name in model_names},
        "per_model": per_model,
        "baseline": {
            "mean_auc": float(baseline_values.mean()) if baseline_values.size else None,
            "std_auc": float(baseline_values.std()) if baseline_values.size else None,
            "fold_aucs": baseline_values.round(6).tolist(),
        },
        "ensemble": {
            "mean_auc": float(ensemble_values.mean()),
            "std_auc": float(ensemble_values.std()),
            "fold_aucs": ensemble_values.round(6).tolist(),
            "per_seed": ensemble_per_seed,
        },
        "integrity": integrity,
        "fold_metadata": fold_metadata,
        "oof_rank_by_seed": oof_rank_by_seed,
    }
