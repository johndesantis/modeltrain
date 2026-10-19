"""Base model factories and scoring helpers (step 5).

All models consume the same TF-IDF feature matrix so cross-validation can
build features once per fold and reuse them.  The boosting model additionally
receives the dense safeguard features (``handcrafted_features``).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from step_01_config import BASELINE_PARAMS, MODEL_PARAMS

try:  # LightGBM is a normal dependency but import lazily for a clear message.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - exercised only on broken installs
    LGBMClassifier = None  # type: ignore[assignment]


def _knn_weights(decay: float):
    """Distance -> weight callable for scikit-learn KNN.

    ``exp(-decay * distance)`` is proportional to ``exp(decay * similarity)``
    for cosine distance (distance = 1 - similarity), matching the similarity
    weighting used in the experiments.
    """

    def weights(distances: np.ndarray) -> np.ndarray:
        return np.exp(-decay * distances)

    return weights


def make_model(name: str):
    """Create an unfitted estimator for ``name``."""
    if name not in MODEL_PARAMS:
        raise KeyError(f"Unknown model {name!r}; expected one of {sorted(MODEL_PARAMS)}")

    params = dict(MODEL_PARAMS[name])
    if name == "logreg":
        return LogisticRegression(**params)
    if name == "svc":
        return LinearSVC(**params)
    if name == "complement_nb":
        return ComplementNB(**params)
    if name in {"knn10", "knn20"}:
        decay = params.pop("similarity_decay")
        return KNeighborsClassifier(weights=_knn_weights(decay), algorithm="brute", **params)
    if name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for the 'lgbm' model")
        return LGBMClassifier(**params)
    raise AssertionError(f"Unhandled model {name!r}")


def make_baseline_model() -> LogisticRegression:
    """Simple word-unigram TF-IDF + logistic regression reference model."""
    return LogisticRegression(
        C=BASELINE_PARAMS["C"],
        solver=BASELINE_PARAMS["solver"],
        max_iter=BASELINE_PARAMS["max_iter"],
        random_state=BASELINE_PARAMS["random_state"],
    )


def model_matrix(name: str, X, safeguard_features: np.ndarray | None = None):
    """Return the matrix a model should be fitted on.

    Only LightGBM consumes the dense safeguard block; appending it to the
    sparse TF-IDF matrix lets the tree model use explicit injection patterns
    together with n-gram features.
    """
    if name != "lgbm" or safeguard_features is None:
        return X
    dense = np.asarray(safeguard_features, dtype=np.float64)
    if issparse(X):
        return hstack([X, dense]).tocsr()
    return np.hstack([np.asarray(X), dense])


def predict_scores(model, X) -> np.ndarray:
    """Return a continuous score where larger means "more jailbreak-like"."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X)).ravel()


def fit_and_score(
    name: str,
    X_train,
    y_train: np.ndarray,
    X_score,
    safeguard_train: np.ndarray | None = None,
    safeguard_score: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a fresh model and score ``X_score``."""
    model = make_model(name)
    train_matrix = model_matrix(name, X_train, safeguard_train)
    score_matrix = model_matrix(name, X_score, safeguard_score)
    model.fit(train_matrix, y_train)
    return predict_scores(model, score_matrix)


def model_is_probability_based(name: str) -> bool:
    """Whether ``name`` emits probabilities (used for reporting only)."""
    return name in {"logreg", "complement_nb", "knn10", "knn20", "lgbm"}


def describe_models() -> list[dict[str, object]]:
    """Small table of model names and their key hyper-parameters."""
    rows = []
    for name in MODEL_PARAMS:
        rows.append(
            {
                "model": name,
                "type": type(make_model(name)).__name__,
                "emits_probabilities": model_is_probability_based(name),
            }
        )
    return rows
