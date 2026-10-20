"""
4_model_trainer.py
Trains the jailbreak classifier ensemble.

Design notes
------------
* **Group-aware cross-validation.**  Prompts that are near-duplicates of each
  other (same template, same source, light paraphrase) are clustered into
  groups and always land in the same fold.  Without this safeguard a model can
  memorise a template from the training half and then "predict" its twin in
  the validation half, which inflates the offline ROC-AUC and hides real
  generalisation gaps.
* **Stacking.**  Every base model produces out-of-fold probabilities; a
  logistic meta-learner combines them.  Stacking is fitted on out-of-fold
  predictions only, so the meta-learner never sees a base model's prediction
  for a row that the base model was trained on.
* **Determinism.**  All estimators are seeded, so repeated runs of the
  notebook reproduce identical numbers.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from importlib import import_module

_feature_builder = import_module("3_feature_builder")
_preprocessor = import_module("2_preprocessor")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Groups (near-duplicate clustering) - the leakage safeguard
# ---------------------------------------------------------------------------

def build_near_duplicate_groups(texts: Sequence[str], threshold: float = 0.80) -> np.ndarray:
    """
    Cluster near-duplicate prompts so they never straddle a CV fold boundary.

    Character n-gram cosine similarity is used, which is robust to small edits
    (punctuation, casing, typos) that a normalised-exact-match test would miss.
    Prompts whose similarity is at least ``threshold`` are merged into one
    group via union-find.

    Returns:
        np.ndarray of group ids, one per input prompt.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    normalized = _preprocessor.normalize_series(texts)
    n = len(normalized)
    if n == 0:
        return np.zeros(0, dtype=int)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(normalized)
    matrix = sp.csr_matrix(matrix)

    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n, 50))
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    similarities = 1.0 - distances

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j_pos in range(indices.shape[1]):
            j = int(indices[i, j_pos])
            if j <= i:
                continue
            if similarities[i, j_pos] >= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    # Re-index to 0..k-1 for compactness.
    mapping: Dict[int, int] = {}
    groups = np.zeros(n, dtype=int)
    for i, root in enumerate(roots):
        if root not in mapping:
            mapping[root] = len(mapping)
        groups[i] = mapping[root]
    return groups


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------

def _lr(C: float = 4.0, **kwargs) -> LogisticRegression:
    params = dict(
        C=C,
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def _base_set(class_weight="balanced"):
    """Return the base-model zoo as ``(name, pipeline)`` pairs.

    Every pipeline normalises -> vectorises -> classifies, and is cloned
    before each fold so that no vectorizer state, IDF weight or word-ID
    mapping is ever learned from a validation row.

    Members
    -------
    word_lr        word (1,2)   + LogisticRegression  (lexical)
    char_lr        char_wb(2,6) + LogisticRegression  (character shape)
    char_svm       char_wb(2,6) + LinearSVC           (margin, different errors)
    word_svm       word (1,2)   + LinearSVC           (margin, different errors)
    joint_lr       word(1,2)+char_wb(2,6)             (joint lexical view)
    char_wide_lr   char_wb(2,8)  + LogisticRegression  (longer-range obfuscation)

    The members disagree on different prompts, so the logistic stacker built
    on their out-of-fold predictions earns its lift over the best single
    model.  The two wider/panoramic members capture patterns the narrow (2,6)
    view smooths over.
    """
    from sklearn.svm import LinearSVC

    models: List[Tuple[str, object]] = []

    # 1. Lexical model: word n-grams + logistic regression.
    models.append((
        "word_lr",
        _feature_builder.build_full_pipeline(
            _lr(C=64.0), blocks=("word_tfidf",)
        ),
    ))

    # 2. Character model: char n-grams + logistic regression.
    models.append((
        "char_lr",
        _feature_builder.build_full_pipeline(
            _lr(C=32.0), blocks=("char_tfidf",)
        ),
    ))

    # 3. Character model with a linear SVM margin (different error profile).
    models.append((
        "char_svm",
        _feature_builder.build_full_pipeline(
            LinearSVC(
                C=2.0,
                class_weight=class_weight,
                max_iter=40000,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),
            blocks=("char_tfidf",),
        ),
    ))

    # 4. Word model with a linear SVM margin (different error profile).
    models.append((
        "word_svm",
        _feature_builder.build_full_pipeline(
            LinearSVC(
                C=1.0,
                class_weight=class_weight,
                max_iter=40000,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),
            blocks=("word_tfidf",),
        ),
    ))

    # 5. Joint lexical view: word + char in one feature union.
    models.append((
        "joint_lr",
        _feature_builder.build_full_pipeline(
            _lr(C=32.0), blocks=("word_tfidf", "char_tfidf")
        ),
    ))

    # 6. Wider character view (n-grams 2-8) for longer-range obfuscation.
    models.append((
        "char_wide_lr",
        _feature_builder.build_full_pipeline(
            _lr(C=16.0), blocks=("char_wide_tfidf",)
        ),
    ))

    return models


def build_base_models() -> List[Tuple[str, object]]:
    """Build the base-model zoo (default configuration)."""
    return _base_set()


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    models: Sequence[Tuple[str, object]],
    texts: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run grouped, stratified cross-validation.

    Every base model is cloned before each fold, so no vectorizer state, IDF
    weight or word-ID mapping is ever learned from a validation row.

    Returns:
        oof            : (n_rows, n_models) out-of-fold probabilities
        auc_by_model   : dict model name -> out-of-fold ROC-AUC
    """
    from sklearn.metrics import roc_auc_score

    n_models = len(models)
    oof = np.zeros((len(texts), n_models), dtype=np.float64)
    splits = list(StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    ).split(np.zeros(len(texts)), y, groups))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        fold_texts = [texts[i] for i in train_idx]
        valid_texts = [texts[i] for i in valid_idx]
        for m, (name, pipeline) in enumerate(models):
            est = clone(pipeline)
            est.fit(fold_texts, y[train_idx])
            if hasattr(est, "predict_proba"):
                oof[valid_idx, m] = est.predict_proba(valid_texts)[:, 1]
            else:
                decision = est.decision_function(valid_texts)
                oof[valid_idx, m] = 1.0 / (1.0 + np.exp(-decision))
        print(f"  fold {fold + 1}/{len(splits)} done (train={len(train_idx)}, valid={len(valid_idx)})")

    auc_by_model: Dict[str, float] = {}
    for m, (name, _) in enumerate(models):
        auc_by_model[name] = float(roc_auc_score(y, oof[:, m]))
    return oof, auc_by_model


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

def build_stacker_features(
    base_predictions: np.ndarray,
    style_features: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assemble the meta-learner input matrix.

    Columns 0..k-1 are the out-of-fold base-model probabilities.  When
    ``style_features`` is supplied, the 17 dense structural features
    (length, punctuation, jailbreak-marker counts, ...) are appended.

    The style features are *deterministic functions of the raw text*: they
    involve no fitting, no label information and no shared state between
    train and test.  They therefore carry no leakage risk, and they give the
    stacker access to a structural view of the prompt that a pure bag-of-
    n-grams representation cannot express (e.g. "this prompt contains five
    instruction-override markers" is not a term any n-gram can carry).
    """
    base = np.asarray(base_predictions, dtype=np.float64)
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    if style_features is None:
        return base
    style = np.asarray(style_features, dtype=np.float64)
    assert style.shape[0] == base.shape[0], \
        f"style features have {style.shape[0]} rows but base predictions have {base.shape[0]}"
    return np.hstack([base, style])


def fit_stacker(
    oof: np.ndarray,
    y: np.ndarray,
    style_features: np.ndarray | None = None,
    C: float = 1.0,
) -> LogisticRegression:
    """
    Fit the logistic meta-learner on out-of-fold base-model predictions.

    The meta-learner only ever sees out-of-fold predictions, so it never
    learns from a base model's opinion about a row that base model was
    trained on.
    """
    X = build_stacker_features(oof, style_features)
    stacker = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    stacker.fit(X, y)
    return stacker


def fit_final_models(
    models: Sequence[Tuple[str, object]],
    texts: Sequence[str],
    y: np.ndarray,
) -> List[Tuple[str, object]]:
    """Refit every base model on the full training set."""
    fitted: List[Tuple[str, object]] = []
    for name, pipeline in models:
        est = clone(pipeline)
        est.fit(list(texts), y)
        fitted.append((name, est))
    return fitted


def predict_ensemble(
    fitted_models: Sequence[Tuple[str, object]],
    stacker: LogisticRegression,
    texts: Sequence[str],
    style_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with every base model, then combine with the stacker.

    Returns:
        (stacked_probabilities, base_probabilities)
    """
    base = np.zeros((len(texts), len(fitted_models)), dtype=np.float64)
    for m, (_, est) in enumerate(fitted_models):
        if hasattr(est, "predict_proba"):
            base[:, m] = est.predict_proba(list(texts))[:, 1]
        else:
            base[:, m] = 1.0 / (1.0 + np.exp(-est.decision_function(list(texts))))
    X = build_stacker_features(base, style_features)
    stacked = stacker.predict_proba(X)[:, 1]
    return stacked, base


def encode_labels(label_series) -> Tuple[np.ndarray, LabelEncoder]:
    """Map 'benign' -> 0 and 'jailbreak' -> 1."""
    encoder = LabelEncoder()
    encoder.fit(["benign", "jailbreak"])
    y = encoder.transform(label_series)
    return y.astype(int), encoder
