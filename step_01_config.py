"""Project configuration (step 1 of the numbered helper modules).

Every other module reads its defaults from here so the notebook, the test
runner, and the report all agree on paths, seeds, folds, and model settings.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
SAMPLE_SUBMISSION_PATH = ROOT / "sample_submission.csv"
SUBMISSION_PATH = ROOT / "submission.csv"
REPORT_PATH = ROOT / "training_report.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
# Two independent grouped-CV repeats keep the estimate stable without making
# the notebook slow.  Every split is deterministic for a given seed.
CV_SEEDS = (42, 7)
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Data safeguards
# ---------------------------------------------------------------------------
LABEL_POSITIVE = "jailbreak"
LABEL_NEGATIVE = "benign"

# Prompts at or above this character TF-IDF cosine similarity are treated as
# near duplicates.  They are kept in the same CV group so duplicate/near
# duplicate prompts can never leak across train and validation folds.
NEAR_DUP_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# The feature matrix shared by the linear / neighbours / boosting models:
#   * word n-grams (1-3)          -> jailbreak vocabulary and phrasing
#   * character n-grams (2-6)     -> obfuscation, spacing, punctuation, leetspeak
WORD_NGRAM_RANGE = (1, 3)
CHAR_NGRAM_RANGE = (2, 6)
MIN_DF = 2
SUBLINEAR_TF = True

# Model hyper-parameters were selected with grouped cross-validation (never
# with the test labels, which do not exist in this repository).
MODEL_PARAMS = {
    "logreg": {
        "C": 32.0,
        "solver": "liblinear",
        "max_iter": 5000,
        "random_state": SEED,
    },
    "svc": {
        "C": 1.0,
        "max_iter": 20000,
        "random_state": SEED,
    },
    "complement_nb": {
        "alpha": 0.3,
    },
    "knn10": {
        "n_neighbors": 10,
        "metric": "cosine",
        "similarity_decay": 5.0,
        "n_jobs": -1,
    },
    "knn20": {
        "n_neighbors": 20,
        "metric": "cosine",
        "similarity_decay": 5.0,
        "n_jobs": -1,
    },
    "lgbm": {
        "n_estimators": 400,
        "learning_rate": 0.06,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.3,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": max(1, os.cpu_count() or 1),
        "verbose": -1,
    },
}

# Equal-weight rank fusion.  The weights are fixed *a priori* (not tuned on the
# test set); a small grid search in grouped CV showed this plateau is flat.
MODEL_NAMES = (
    "logreg",
    "svc",
    "complement_nb",
    "knn10",
    "knn20",
    "lgbm",
)
BLEND_WEIGHTS = {name: 1.0 for name in MODEL_NAMES}

# A deliberately simple reference model used only to show the improvement from
# the ensemble.  It is not part of the submission.
BASELINE_PARAMS = {
    "word_ngram_range": (1, 1),
    "min_df": 2,
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 5000,
    "random_state": SEED,
}

# Calibration keeps exported scores in [0, 1] and roughly probability-like.
CALIBRATION_EPS = 1e-6
SUBMISSION_SCORE_DECIMALS = 6


def seed_everything(seed: int = SEED) -> None:
    """Seed Python and NumPy for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything()
