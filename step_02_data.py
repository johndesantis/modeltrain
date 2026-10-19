"""Data loading and schema safeguards (step 2).

The functions here fail loudly when a CSV violates an assumption the model
relies on: missing columns, duplicate / missing ids, unknown labels, a
single-class training file, or a submission that does not line up with the
test ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from step_01_config import LABEL_NEGATIVE, LABEL_POSITIVE

TRAIN_COLUMNS = ("Id", "text", "label")
TEST_COLUMNS = ("Id", "text")
SUBMISSION_COLUMNS = ("Id", "TARGET")

_LABEL_MAP = {
    "jailbreak": 1,
    "jailbroken": 1,
    "malicious": 1,
    "unsafe": 1,
    "attack": 1,
    "1": 1,
    "true": 1,
    "yes": 1,
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "0": 0,
    "false": 0,
    "no": 0,
}


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a competition CSV, keeping ids as integers when possible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df["Id"] = pd.to_numeric(df["Id"], errors="raise").astype("int64")
    return df


def parse_label(value: Any) -> int:
    """Map one label value to 0 (benign) or 1 (jailbreak).

    Unknown labels raise ``ValueError`` instead of being silently coerced.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return int(value)
        raise ValueError(f"Unsupported integer label: {value!r}")
    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return int(value)
        raise ValueError(f"Unsupported float label: {value!r}")
    key = str(value).strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    raise ValueError(f"Unsupported label: {value!r}")


def parse_labels(values: Iterable[Any]) -> np.ndarray:
    """Vectorised version of :func:`parse_label`."""
    return np.asarray([parse_label(v) for v in values], dtype=np.int64)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique_ids(df: pd.DataFrame, name: str) -> None:
    if df["Id"].isna().any():
        raise ValueError(f"{name} contains missing Id values")
    duplicates = df["Id"][df["Id"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{name} contains duplicate Id values: {duplicates[:10]}")


def validate_train(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the training frame and return a JSON-serialisable summary."""
    _require_columns(df, TRAIN_COLUMNS, "train.csv")
    _require_unique_ids(df, "train.csv")
    if df["text"].isna().any():
        raise ValueError("train.csv contains missing text values")
    if df["label"].isna().any():
        raise ValueError("train.csv contains missing labels")

    labels = parse_labels(df["label"])
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        raise ValueError(
            "train.csv must contain both benign and jailbreak examples; "
            f"found counts {dict(zip(('benign', 'jailbreak'), counts.tolist()))}"
        )

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "class_counts": {
            LABEL_NEGATIVE: int(counts[0]),
            LABEL_POSITIVE: int(counts[1]),
        },
        "positive_rate": float(counts[1] / counts.sum()),
        "unique_ids": int(df["Id"].nunique()),
        "empty_texts": int((df["text"].astype(str).str.strip() == "").sum()),
        "mean_text_chars": float(df["text"].astype(str).str.len().mean()),
    }


def validate_test(
    df: pd.DataFrame,
    sample_submission: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate the test frame (and its match with the sample submission)."""
    _require_columns(df, TEST_COLUMNS, "test.csv")
    _require_unique_ids(df, "test.csv")
    if df["text"].isna().any():
        raise ValueError("test.csv contains missing text values")

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "empty_texts": int((df["text"].astype(str).str.strip() == "").sum()),
        "mean_text_chars": float(df["text"].astype(str).str.len().mean()),
    }

    if sample_submission is not None:
        _require_columns(sample_submission, SUBMISSION_COLUMNS, "sample_submission.csv")
        sample_ids = set(sample_submission["Id"].astype("int64").tolist())
        test_ids = set(df["Id"].tolist())
        if sample_ids != test_ids:
            missing = sorted(sample_ids - test_ids)[:10]
            extra = sorted(test_ids - sample_ids)[:10]
            raise ValueError(
                "test.csv and sample_submission.csv ids differ; "
                f"missing from test={missing}, extra in test={extra}"
            )
        summary["sample_submission_rows"] = int(len(sample_submission))
        summary["ids_match_sample_submission"] = True
        summary["sample_submission_order_matches_test"] = bool(
            sample_submission["Id"].astype("int64").tolist() == df["Id"].tolist()
        )
    return summary


def check_disjoint_ids(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Ensure no prompt id is present in both files."""
    overlap = sorted(set(train["Id"]).intersection(set(test["Id"])))
    if overlap:
        raise ValueError(f"train.csv and test.csv share ids: {overlap[:10]}")
    return {"overlapping_ids": 0}


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write JSON atomically (temporary file + replace)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    tmp.replace(path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
