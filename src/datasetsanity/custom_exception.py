from __future__ import annotations

from typing import Optional, Sequence, Tuple


class DatasetSanityError(Exception):
    """Base exception class for DatasetSanity errors."""


class MissingValuesError(DatasetSanityError):
    """Raised when the dataset contains unexpected missing values."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.columns: Optional[Tuple[str, ...]] = tuple(columns) if columns else None

        if self.columns:
            message = message or (
                f"Missing values detected in columns: {', '.join(self.columns)}"
            )
        else:
            message = message or "Missing values detected in dataset."

        super().__init__(message)


class ClassImbalanceError(DatasetSanityError):
    """Raised when the dataset has a severe class imbalance."""

    def __init__(
        self,
        target_column: Optional[str] = None,
        imbalance_ratio: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        self.target_column = target_column
        self.imbalance_ratio = imbalance_ratio

        if target_column is not None and imbalance_ratio is not None:
            message = message or (
                f"Severe class imbalance detected in '{target_column}' "
                f"(imbalance ratio: {imbalance_ratio:.2f})"
            )
        else:
            message = message or "Severe class imbalance detected in dataset."

        super().__init__(message)


class DataLeakageError(DatasetSanityError):
    """Raised when potential data leakage is detected."""

    def __init__(
        self,
        features: Optional[Sequence[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.features: Optional[Tuple[str, ...]] = tuple(features) if features else None

        if self.features:
            message = message or (
                f"Potential data leakage detected in features: {', '.join(self.features)}"
            )
        else:
            message = message or "Potential data leakage detected in dataset."

        super().__init__(message)