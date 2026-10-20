"""
6_submission_exporter.py
Writes the leaderboard-ready ``submission.csv``.

Schema (mandated by the issue):

    Id      - the prompt id, taken verbatim from ``test.csv``
    TARGET  - the jailbreak likelihood score in [0, 1].
              TARGET >= 0.5 -> "jailbreak", TARGET < 0.5 -> "benign"

The exporter re-validates the produced frame against ``sample_submission.csv``
before writing, so a schema drift (renamed column, missing row, wrong order)
fails loudly instead of producing a silently unscored submission.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd


def build_submission(
    test_ids,
    scores: np.ndarray,
    sample_submission_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assemble and validate the submission frame.

    Parameters
    ----------
    test_ids : sequence
        Ids from ``test.csv``, in the original order.
    scores : array-like
        Jailbreak likelihood in [0, 1], aligned with ``test_ids``.
    sample_submission_path : str, optional
        When given, the frame is checked against this file for column names,
        row count and id set.

    Returns:
        pd.DataFrame with columns ['Id', 'TARGET'].
    """
    ids = pd.Series(list(test_ids)).reset_index(drop=True)
    target = pd.Series(np.asarray(scores, dtype=float)).reset_index(drop=True)

    assert len(ids) == len(target), \
        f"Id/score length mismatch: {len(ids)} ids vs {len(target)} scores"
    assert not ids.isna().any(), "Submission contains missing Ids"
    assert ids.is_unique, "Submission contains duplicate Ids"
    assert np.isfinite(target.to_numpy()).all(), "Submission contains NaN or infinite scores"
    assert (target >= 0.0).all() and (target <= 1.0).all(), \
        "TARGET scores must lie in [0, 1]"

    submission = pd.DataFrame({"Id": ids, "TARGET": target})

    if sample_submission_path and os.path.exists(sample_submission_path):
        sample = pd.read_csv(sample_submission_path, dtype={"Id": int, "TARGET": float})
        assert list(sample.columns) == ["Id", "TARGET"], \
            f"Sample submission columns are {list(sample.columns)}, expected ['Id', 'TARGET']"
        assert len(sample) == len(submission), \
            f"Row count mismatch: sample has {len(sample)}, submission has {len(submission)}"
        assert set(sample["Id"]) == set(submission["Id"]), \
            "Submission Ids do not match the sample submission Ids"
        # Preserve the sample submission's row order.
        order = pd.Series(sample["Id"].values, index=sample.index)
        submission = (
            submission.set_index("Id")
            .reindex(order.values)
            .reset_index()
            .rename(columns={"index": "Id"})
        )
        assert submission["TARGET"].notna().all(), "Re-ordering produced missing TARGET values"

    return submission


def write_submission(submission: pd.DataFrame, output_path: str) -> str:
    """Write the submission to ``output_path`` and return the path."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def summarize_submission(submission: pd.DataFrame) -> str:
    """Human readable summary of the produced submission."""
    target = submission["TARGET"].to_numpy(dtype=float)
    n_jailbreak = int((target >= 0.5).sum())
    n_benign = int((target < 0.5).sum())
    lines = [
        f"Rows            : {len(submission)}",
        f"Columns         : {list(submission.columns)}",
        f"Id range        : {submission['Id'].min()} .. {submission['Id'].max()}",
        f"TARGET min/max  : {target.min():.6f} / {target.max():.6f}",
        f"TARGET mean     : {target.mean():.6f}",
        f"Labeled jailbreak (>=0.5): {n_jailbreak} ({n_jailbreak / max(1, len(submission)):.2%})",
        f"Labeled benign   (<0.5) : {n_benign} ({n_benign / max(1, len(submission)):.2%})",
    ]
    return "\n".join(lines)


def verify_submission_file(path: str, expected_rows: Optional[int] = None) -> bool:
    """Re-read the written file from disk and assert it is well formed."""
    frame = pd.read_csv(path, dtype={"Id": int, "TARGET": float})
    assert list(frame.columns) == ["Id", "TARGET"], \
        f"Written file has columns {list(frame.columns)}, expected ['Id', 'TARGET']"
    assert frame["Id"].is_unique, "Written file has duplicate Ids"
    assert frame["TARGET"].between(0.0, 1.0).all(), "Written file has out-of-range TARGET values"
    assert frame["TARGET"].notna().all(), "Written file has missing TARGET values"
    if expected_rows is not None:
        assert len(frame) == expected_rows, \
            f"Written file has {len(frame)} rows, expected {expected_rows}"
    return True
