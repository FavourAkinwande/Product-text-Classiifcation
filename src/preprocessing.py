"""
Shared preprocessing pipeline for Amazon product title classification.

This script:
- Loads the raw dataset
- Cleans the text into a `clean_text` column (keeping numbers)
- Encodes labels into integer IDs
- Performs a stratified train/test split
- Saves processed CSVs, label mappings, and metadata for reproducible experiments.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =========================
# Configuration
# =========================

DATA_PATH = "../data/titles_to_categories.csv"

# Use existing dataset column names from the EDA notebook
TEXT_COL = "title"
LABEL_COL = "category_name"

# Set to an integer to sample, or None to load full dataset
NROWS: int | None = 500_000

TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42

RESULTS_DIR = "../results"

# Optional: threshold for "rare" classes, used only for reporting
RARE_CLASS_THRESHOLD: int = 20


@dataclass
class PreprocessingMetadata:
    """Simple container for preprocessing configuration and dataset summary."""

    data_path: str
    text_col: str
    label_col: str
    nrows: int | None
    test_size: float
    random_seed: int
    num_classes: int
    total_rows_loaded: int
    total_rows_after_cleaning: int
    train_size: int
    test_size_rows: int
    rare_class_threshold: int
    num_rare_classes: int
    created_at_utc: str


def ensure_results_dir(path: str) -> None:
    """Create the results directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str, text_col: str, label_col: str, nrows: int | None) -> pd.DataFrame:
    """Load the dataset and keep only the text and label columns, dropping missing rows."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, nrows=nrows)
    print(f"Raw shape: {df.shape}")

    missing_cols = [c for c in (text_col, label_col) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in CSV: {missing_cols}")

    df = df[[text_col, label_col]].copy()
    before_dropna = df.shape[0]
    df = df.dropna(subset=[text_col, label_col])
    print(f"Dropped {before_dropna - df.shape[0]} rows with missing text/label.")
    print(f"Shape after column selection + dropna: {df.shape}")
    return df


_CLEAN_REGEX = re.compile(r"[^a-zA-Z0-9\s]+")


def clean_text(text: Any) -> str:
    """
    Clean a single product title.

    Steps:
    - Convert to lowercase for case-insensitive modeling.
    - Remove punctuation and symbols, but KEEP letters and digits.
    - Collapse multiple spaces into a single space and strip edges.

    We explicitly KEEP numbers because Amazon product titles contain
    structured tokens such as model numbers, sizes, capacities, and counts
    (e.g., "128GB", "iPhone 15", "pack of 3"). These numeric patterns
    are often highly predictive of the correct product category, so
    removing them would discard important signal for classification.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.lower()
    text = _CLEAN_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_clean_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Create `clean_text` column and drop rows that become empty."""
    print("Cleaning text column...")
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)

    before_empty = df.shape[0]
    df = df[df["clean_text"].str.len() > 0]
    print(f"Dropped {before_empty - df.shape[0]} rows that became empty after cleaning.")
    print(f"Shape after cleaning: {df.shape}")
    return df


def encode_labels(df: pd.DataFrame, label_col: str, results_dir: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode the target column and save label mapping to CSV."""
    print("Encoding labels...")
    df = df.copy()

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df[label_col].astype(str))

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Save mapping: original category -> label_id
    mapping_df = pd.DataFrame(
        {
            "category": le.classes_,
            "label_id": np.arange(num_classes, dtype=int),
        }
    )
    mapping_path = os.path.join(results_dir, "label_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved label mapping to: {mapping_path}")

    return df, le


def report_class_distribution(df: pd.DataFrame, label_col: str) -> Tuple[int, int]:
    """Print basic class distribution stats and return (num_classes, num_rare_classes)."""
    counts = df[label_col].value_counts()
    num_classes = counts.shape[0]
    print("\nTop 10 classes by count:")
    print(counts.head(10))

    rare_mask = counts < RARE_CLASS_THRESHOLD
    num_rare_classes = int(rare_mask.sum())
    print(f"\nClasses with < {RARE_CLASS_THRESHOLD} samples: {num_rare_classes} / {num_classes}")
    return num_classes, num_rare_classes


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split using `label_id`."""
    if "label_id" not in df.columns:
        raise KeyError("Column 'label_id' not found. Run label encoding first.")

    X = df.index.values
    y = df["label_id"].values

    train_idx, test_idx = train_test_split(
        X,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    return train_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str,
    label_col: str,
) -> None:
    """
    Save train/test splits to CSV.

    Columns saved:
    - clean_text
    - label_id
    - original label column (e.g., category_name)
    """
    cols_to_keep = ["clean_text", "label_id", label_col]
    train_out = os.path.join(results_dir, "train.csv")
    test_out = os.path.join(results_dir, "test.csv")

    train_df[cols_to_keep].to_csv(train_out, index=False)
    test_df[cols_to_keep].to_csv(test_out, index=False)

    print(f"\nSaved train split to: {train_out}")
    print(f"Saved test split to:  {test_out}")


def save_metadata(
    meta: PreprocessingMetadata,
    results_dir: str,
) -> None:
    """Save preprocessing metadata as JSON."""
    meta_path = os.path.join(results_dir, "preprocessing_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"Saved preprocessing metadata to: {meta_path}")


def main() -> None:
    ensure_results_dir(RESULTS_DIR)

    df = load_dataset(DATA_PATH, TEXT_COL, LABEL_COL, NROWS)
    total_rows_loaded = df.shape[0]

    num_classes_raw, num_rare_classes = report_class_distribution(df, LABEL_COL)

    df = add_clean_text_column(df, TEXT_COL)
    total_rows_after_cleaning = df.shape[0]

    df_encoded, _ = encode_labels(df, LABEL_COL, RESULTS_DIR)

    train_df, test_df = stratified_split(df_encoded, TEST_SIZE, RANDOM_SEED)

    save_splits(train_df, test_df, RESULTS_DIR, LABEL_COL)

    meta = PreprocessingMetadata(
        data_path=DATA_PATH,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        nrows=NROWS,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        num_classes=num_classes_raw,
        total_rows_loaded=total_rows_loaded,
        total_rows_after_cleaning=total_rows_after_cleaning,
        train_size=train_df.shape[0],
        test_size_rows=test_df.shape[0],
        rare_class_threshold=RARE_CLASS_THRESHOLD,
        num_rare_classes=num_rare_classes,
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    save_metadata(meta, RESULTS_DIR)


if __name__ == "__main__":
    main()

