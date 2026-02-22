from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from datasetsanity.custom_exception import (
    MissingValuesError,
    ClassImbalanceError,
    DataLeakageError,
)


def check_missing_values(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> None:
    """
    Raise MissingValuesError if missing values are detected.
    """
    if columns is None:
        columns = df.columns

    missing_cols = [
        col for col in columns if df[col].isnull().any()
    ]

    if missing_cols:
        raise MissingValuesError(columns=missing_cols)


def check_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.9,
) -> None:
    """
    Raise ClassImbalanceError if the dominant class ratio exceeds threshold.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    class_ratios = df[target_column].value_counts(normalize=True)

    max_ratio = class_ratios.max()
    if max_ratio >= threshold:
        raise ClassImbalanceError(
            target_column=target_column,
            imbalance_ratio=float(max_ratio),
        )


def check_data_leakage(
    df: pd.DataFrame,
    target_column: str,
    correlation_threshold: float = 0.95,
) -> None:
    """
    Detect features overly correlated with target.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    numeric_df = df.select_dtypes(include="number")

    if target_column not in numeric_df.columns:
        return

    correlations = numeric_df.corr()[target_column].abs()

    leaked_features = [
        col for col, corr in correlations.items()
        if col != target_column and corr >= correlation_threshold
    ]

    if leaked_features:
        raise DataLeakageError(features=leaked_features)