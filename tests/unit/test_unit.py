import pytest
from datasetsanity.custom_exception import (
    MissingValuesError,
    ClassImbalanceError,
    DataLeakageError,
)


def test_missing_values_error_with_columns():
    exc = MissingValuesError(columns=["col1", "col2"])
    assert "col1" in str(exc)
    assert "col2" in str(exc)


def test_missing_values_error_without_columns():
    exc = MissingValuesError()
    assert "Missing values detected in dataset" in str(exc)


def test_class_imbalance_error():
    exc = ClassImbalanceError("label", 0.92)
    assert "label" in str(exc)
    assert "0.92" in str(exc)


def test_data_leakage_error():
    exc = DataLeakageError(features=["target", "leak_feature"])
    assert "target" in str(exc)
    assert "leak_feature" in str(exc)
    
