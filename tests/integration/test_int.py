import pandas as pd
import pytest

from datasetsanity.custom_exception import (
    MissingValuesError,
    ClassImbalanceError,
    DataLeakageError,
)

# These imports assume future implementation
from datasetsanity.validators import (
    check_missing_values,
    check_class_imbalance,
    check_data_leakage,
)


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