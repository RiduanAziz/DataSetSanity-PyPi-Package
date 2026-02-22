import json
import os

import pandas as pd
import pytest

from datasetsanity.custom_exception import (
    ClassImbalanceError,
    DataLeakageError,
    MissingValuesError,
)
from datasetsanity.core import DatasetSanity, SanityReport, CheckResult
from datasetsanity.validators import (
    check_class_imbalance,
    check_data_leakage,
    check_missing_values,
)


# ---------------------------------------------------------------------------
# Existing exception tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

def test_check_missing_values_passes_on_clean_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    check_missing_values(df)  # should not raise


def test_check_missing_values_raises_on_nan():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    with pytest.raises(MissingValuesError) as exc_info:
        check_missing_values(df)
    assert exc_info.value.columns is not None
    assert "a" in exc_info.value.columns


def test_check_missing_values_records_affected_columns():
    df = pd.DataFrame({"x": [None, 2], "y": [1, None], "z": [1, 2]})
    with pytest.raises(MissingValuesError) as exc_info:
        check_missing_values(df)
    assert "x" in exc_info.value.columns
    assert "y" in exc_info.value.columns
    assert "z" not in exc_info.value.columns


# ---------------------------------------------------------------------------
# check_class_imbalance
# ---------------------------------------------------------------------------

def test_check_class_imbalance_passes_on_balanced():
    df = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1]})
    check_class_imbalance(df, target_column="label", threshold=0.9)  # should not raise


def test_check_class_imbalance_raises_on_dominant_class():
    df = pd.DataFrame({"label": [1] * 95 + [0] * 5})
    with pytest.raises(ClassImbalanceError):
        check_class_imbalance(df, target_column="label", threshold=0.9)


def test_check_class_imbalance_stores_target_and_ratio():
    df = pd.DataFrame({"label": [1] * 95 + [0] * 5})
    with pytest.raises(ClassImbalanceError) as exc_info:
        check_class_imbalance(df, target_column="label", threshold=0.9)
    assert exc_info.value.target_column == "label"
    assert exc_info.value.imbalance_ratio is not None
    assert exc_info.value.imbalance_ratio >= 0.9


# ---------------------------------------------------------------------------
# check_data_leakage
# ---------------------------------------------------------------------------

def test_check_data_leakage_passes_when_no_perfect_correlation():
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    check_data_leakage(df, target_column="target", correlation_threshold=0.95)  # should not raise


def test_check_data_leakage_raises_on_perfect_correlation():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "leak": [0, 1, 0, 1]})
    with pytest.raises(DataLeakageError):
        check_data_leakage(df, target_column="target", correlation_threshold=0.95)


# ---------------------------------------------------------------------------
# SanityReport.summary()
# ---------------------------------------------------------------------------

def _make_report(mv_passed=True, ci_passed=True, lk_passed=True):
    return SanityReport(
        missing_values=CheckResult(passed=mv_passed, details={}),
        class_imbalance=CheckResult(passed=ci_passed, details={}),
        leakage=CheckResult(passed=lk_passed, details={}),
    )


def test_sanity_report_summary_all_pass(capsys):
    report = _make_report()
    report.summary()
    captured = capsys.readouterr()
    assert "\u2714" in captured.out  # ✔
    assert "PASSED" in captured.out


def test_sanity_report_summary_with_failure(capsys):
    report = _make_report(mv_passed=False)
    report.summary()
    captured = capsys.readouterr()
    assert "\u274c" in captured.out  # ❌
    assert "FAILED" in captured.out
    assert "\u26a0" in captured.out  # ⚠


# ---------------------------------------------------------------------------
# SanityReport.to_json()
# ---------------------------------------------------------------------------

def test_sanity_report_to_json_writes_valid_json(tmp_path):
    report = _make_report()
    output_path = str(tmp_path / "report.json")
    report.to_json(output_path)
    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
    assert "missing_values" in data
    assert "class_imbalance" in data
    assert "leakage" in data
    assert data["missing_values"]["passed"] is True


# ---------------------------------------------------------------------------
# DatasetSanity.run()
# ---------------------------------------------------------------------------

def test_dataset_sanity_run_returns_report():
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    checker = DatasetSanity(df, target="target")
    result = checker.run()
    assert isinstance(result, SanityReport)


def test_dataset_sanity_run_passes_on_clean_df():
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    checker = DatasetSanity(df, target="target")
    report = checker.run()
    assert report.missing_values.passed is True
    assert report.class_imbalance.passed is True
    assert report.leakage.passed is True


def test_dataset_sanity_run_records_missing_values_failure():
    df = pd.DataFrame({"feat": [1, None, 3], "target": [0, 1, 0]})
    checker = DatasetSanity(df, target="target")
    report = checker.run()
    assert report.missing_values.passed is False


def test_dataset_sanity_run_records_imbalance_failure():
    df = pd.DataFrame({"feat": list(range(100)), "target": [1] * 95 + [0] * 5})
    checker = DatasetSanity(df, target="target", imbalance_threshold=0.9)
    report = checker.run()
    assert report.class_imbalance.passed is False


def test_dataset_sanity_run_records_leakage_failure():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "leak": [0, 1, 0, 1]})
    checker = DatasetSanity(df, target="target", correlation_threshold=0.95)
    report = checker.run()
    assert report.leakage.passed is False

