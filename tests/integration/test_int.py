import json
import os

import pandas as pd
import pytest

from datasetsanity.core import DatasetSanity, SanityReport
from datasetsanity.custom_exception import (
    ClassImbalanceError,
    DataLeakageError,
    MissingValuesError,
)
from datasetsanity.validators import (
    check_class_imbalance,
    check_data_leakage,
    check_missing_values,
)


# ---------------------------------------------------------------------------
# Existing validator integration tests
# ---------------------------------------------------------------------------

def test_integration_missing_values_detection():
    df = pd.DataFrame(
        {
            "age": [25, None, 30],
            "salary": [50000, 60000, None],
        }
    )

    with pytest.raises(MissingValuesError) as exc:
        check_missing_values(df)

    assert "age" in str(exc.value)
    assert "salary" in str(exc.value)


def test_integration_class_imbalance_detection():
    df = pd.DataFrame(
        {
            "label": [1] * 95 + [0] * 5
        }
    )

    with pytest.raises(ClassImbalanceError) as exc:
        check_class_imbalance(df, target_column="label", threshold=0.9)

    assert "label" in str(exc.value)


def test_integration_data_leakage_detection():
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "target": [0, 1, 0],
            "leak_feature": [0, 1, 0],  # perfectly correlated
        }
    )

    with pytest.raises(DataLeakageError) as exc:
        check_data_leakage(
            df,
            target_column="target",
            correlation_threshold=0.99,
        )

    assert "leak_feature" in str(exc.value)


# ---------------------------------------------------------------------------
# DatasetSanity end-to-end flows
# ---------------------------------------------------------------------------

def test_full_run_clean_df_all_pass():
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    report = DatasetSanity(df, target="target").run()
    assert report.missing_values.passed is True
    assert report.class_imbalance.passed is True
    assert report.leakage.passed is True


def test_full_run_with_missing_values_fails():
    df = pd.DataFrame({"feat": [1, None, 3], "target": [0, 1, 0]})
    report = DatasetSanity(df, target="target").run()
    assert report.missing_values.passed is False


def test_full_run_with_imbalanced_df_fails():
    df = pd.DataFrame({"feat": list(range(100)), "target": [1] * 95 + [0] * 5})
    report = DatasetSanity(df, target="target", imbalance_threshold=0.9).run()
    assert report.class_imbalance.passed is False


def test_full_run_with_leakage_fails():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "leak": [0, 1, 0, 1]})
    report = DatasetSanity(df, target="target", correlation_threshold=0.95).run()
    assert report.leakage.passed is False


def test_to_json_integration(tmp_path):
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    report = DatasetSanity(df, target="target").run()
    output_path = str(tmp_path / "report.json")
    report.to_json(output_path)
    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
    assert data["missing_values"]["passed"] is True
    assert data["class_imbalance"]["passed"] is True
    assert data["leakage"]["passed"] is True


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

def test_cli_check_clean_csv(tmp_path):
    from click.testing import CliRunner
    from datasetsanity.cli import main

    csv_path = str(tmp_path / "data.csv")
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(main, ["check", csv_path, "--target", "target"])
    assert result.exit_code == 0
    assert "PASSED" in result.output


def test_cli_check_missing_values_exits_1(tmp_path):
    from click.testing import CliRunner
    from datasetsanity.cli import main

    csv_path = str(tmp_path / "data.csv")
    df = pd.DataFrame({"feat": [1, None, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(main, ["check", csv_path, "--target", "target"])
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_cli_check_writes_output_json(tmp_path):
    from click.testing import CliRunner
    from datasetsanity.cli import main

    csv_path = str(tmp_path / "data.csv")
    report_path = str(tmp_path / "report.json")
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(main, ["check", csv_path, "--target", "target", "--output", report_path])
    assert result.exit_code == 0
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert "missing_values" in data
