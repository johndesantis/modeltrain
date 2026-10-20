"""
1_data_loader.py
Handles loading and validation of train/test data.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSV files.
    
    Returns:
        train_df: DataFrame with columns ['Id', 'text', 'label']
        test_df: DataFrame with columns ['Id', 'text']
    """
    train_df = pd.read_csv(train_path, dtype={'Id': int, 'text': str, 'label': str})
    test_df = pd.read_csv(test_path, dtype={'Id': int, 'text': str})
    
    # Ensure required columns exist
    assert 'Id' in train_df.columns and 'text' in train_df.columns and 'label' in train_df.columns, \
        "Train CSV must have columns: Id, text, label"
    assert 'Id' in test_df.columns and 'text' in test_df.columns, \
        "Test CSV must have columns: Id, text"
    
    return train_df, test_df


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Validate data integrity: no leakage, clean split, consistent Id ranges.
    Raises AssertionError if validation fails.
    """
    # Basic structure checks
    assert train_df['label'].isin(['jailbreak', 'benign']).all(), \
        "Train labels must be only 'jailbreak' or 'benign'"
    assert not train_df['text'].isna().any(), "Train text has missing values"
    assert not test_df['text'].isna().any(), "Test text has missing values"
    
    # Id uniqueness
    assert train_df['Id'].is_unique, "Train Ids must be unique"
    assert test_df['Id'].is_unique, "Test Ids must be unique"
    assert len(set(train_df['Id']) & set(test_df['Id'])) == 0, \
        "Train and test Ids must not overlap"
    
    # Text normalization for leakage check
    def normalize_text(s: str) -> str:
        if pd.isna(s):
            return ""
        # Lowercase, strip, collapse whitespace
        s = str(s).strip().lower()
        s = re.sub(r'\s+', ' ', s)
        return s
    
    train_norm = set(train_df['text'].apply(normalize_text))
    test_norm = set(test_df['text'].apply(normalize_text))
    overlap = train_norm & test_norm
    assert len(overlap) == 0, \
        f"Found {len(overlap)} overlapping normalized texts between train and test. " \
        "This indicates data leakage."
    
    # Check label distribution sanity
    label_counts = train_df['label'].value_counts()
    print(f"Label distribution - benign: {label_counts.get('benign', 0)}, jailbreak: {label_counts.get('jailbreak', 0)}")
    
    # Id range sanity (should be reasonable integers)
    assert train_df['Id'].between(1, 1000000).all(), "Train Ids out of expected range"
    assert test_df['Id'].between(1, 1000000).all(), "Test Ids out of expected range"
    
    print("✓ Data validation passed: clean train/test split, no leakage, valid Ids and labels")


def get_submission_schema(sample_submission_path: str) -> pd.DataFrame:
    """
    Load sample submission to verify expected format.
    
    Returns:
        DataFrame with columns ['Id', 'TARGET']
    """
    ss_df = pd.read_csv(sample_submission_path, dtype={'Id': int, 'TARGET': float})
    assert set(ss_df.columns) == {'Id', 'TARGET'}, \
        "Sample submission must have columns: Id, TARGET"
    assert ss_df['TARGET'].between(0, 1).all(), \
        "TARGET values must be between 0 and 1"
    return ss_df[['Id']].copy()  # Return Id column only for filling predictions