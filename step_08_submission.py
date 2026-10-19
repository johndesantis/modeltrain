"""Submission construction and validation (step 8).

The exported file contains exactly ``Id,TARGET`` in the order of the supplied
test ids.  ``TARGET`` is the calibrated jailbreak likelihood in ``[0, 1]``;
values >= 0.5 correspond to the binary label "jailbreak".
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from step_01_config import SUBMISSION_SCORE_DECIMALS


def build_submission(
    ids: Sequence[int],
    scores: Sequence[float],
    decimals: int = SUBMISSION_SCORE_DECIMALS,
) -> pd.DataFrame:
    """Build the ``Id,TARGET`` frame and round the scores."""
    ids = np.asarray(ids, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if ids.size != scores.size:
        raise ValueError(f"ids ({ids.size}) and scores ({scores.size}) differ in length")
    submission = pd.DataFrame(
        {
            "Id": ids,
            "TARGET": np.round(scores, decimals),
        }
    )
    return submission


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: Sequence[int],
    score_column: str = "TARGET",
) -> dict[str, object]:
    """Fail loudly if the submission violates the competition contract."""
    if list(submission.columns) != ["Id", score_column]:
        raise ValueError(
            f"submission columns must be ['Id', '{score_column}'], "
            f"got {list(submission.columns)}"
        )
    expected_ids = np.asarray(expected_ids, dtype=np.int64).ravel()
    if len(submission) != expected_ids.size:
        raise ValueError(
            f"submission has {len(submission)} rows, expected {expected_ids.size}"
        )
    ids = submission["Id"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != ids.size:
        raise ValueError("submission contains duplicate Id values")
    if not np.array_equal(ids, expected_ids):
        raise ValueError("submission Id order does not match the test Id order")

    scores = submission[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("submission contains NaN or infinite TARGET values")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("TARGET values must lie in [0, 1]")
    if np.unique(scores).size < 2:
        raise ValueError("TARGET is constant; a ranking metric cannot be meaningful")
    return {
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "ids_match_test_order": True,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "fraction_at_or_above_0.5": float((scores >= 0.5).mean()),
        "unique_scores": int(np.unique(scores).size),
    }


def write_submission(
    submission: pd.DataFrame,
    expected_ids: Sequence[int],
    path: str | Path,
    score_column: str = "TARGET",
) -> dict[str, object]:
    """Validate, then write the CSV atomically and return validation stats."""
    stats = validate_submission(submission, expected_ids, score_column=score_column)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    submission.to_csv(tmp, index=False, float_format=f"%.{SUBMISSION_SCORE_DECIMALS}f")
    tmp.replace(path)
    stats["path"] = str(path)
    return stats


def predict_binary_labels(scores: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    """Convenience helper described in the task (>= 0.5 -> jailbreak)."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.where(scores >= threshold, "jailbreak", "benign")
