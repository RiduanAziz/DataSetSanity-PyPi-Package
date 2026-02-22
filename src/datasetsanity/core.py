from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from datasetsanity.custom_exception import (
    ClassImbalanceError,
    DataLeakageError,
    DatasetSanityError,
    MissingValuesError,
)
from datasetsanity.logger import get_logger
from datasetsanity.validators import (
    check_class_imbalance,
    check_data_leakage,
    check_missing_values,
)

logger = get_logger(__name__)

_PASS = "\u2714"
_WARN = "\u26a0"
_FAIL = "\u274c"


class CheckResult:
    """Stores the outcome of a single sanity check."""

    def __init__(self, passed: bool, details: Dict[str, Any]) -> None:
        self.passed = passed
        self.details = details


class SanityReport:
    """Aggregates the results of all three DatasetSanity checks."""

    def __init__(
        self,
        missing_values: CheckResult,
        class_imbalance: CheckResult,
        leakage: CheckResult,
    ) -> None:
        self.missing_values = missing_values
        self.class_imbalance = class_imbalance
        self.leakage = leakage

    def _symbol(self, passed: bool) -> str:
        return _PASS if passed else _FAIL

    def summary(self) -> None:
        """Print a human-readable console report."""
        print("DatasetSanity Report")
        print("=" * 40)

        for name, result in [
            ("Missing Values", self.missing_values),
            ("Class Imbalance", self.class_imbalance),
            ("Data Leakage", self.leakage),
        ]:
            symbol = self._symbol(result.passed)
            status = "PASSED" if result.passed else "FAILED"
            print(f"  {symbol}  {name}: {status}")
            if result.details:
                for key, value in result.details.items():
                    print(f"       {key}: {value}")

        print("=" * 40)
        overall = self.missing_values.passed and self.class_imbalance.passed and self.leakage.passed
        overall_symbol = _PASS if overall else _WARN
        overall_status = "All checks passed" if overall else "Some checks failed"
        print(f"  {overall_symbol}  Overall: {overall_status}")

    def to_json(self, path: str) -> None:
        """Write the report results to a JSON file."""
        data = {
            "missing_values": {
                "passed": self.missing_values.passed,
                "details": self.missing_values.details,
            },
            "class_imbalance": {
                "passed": self.class_imbalance.passed,
                "details": self.class_imbalance.details,
            },
            "leakage": {
                "passed": self.leakage.passed,
                "details": self.leakage.details,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Report written to %s", path)


class DatasetSanity:
    """Orchestrates all dataset sanity checks."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        task: str = "classification",
        imbalance_threshold: float = 0.9,
        correlation_threshold: float = 0.95,
    ) -> None:
        self.df = df
        self.target = target
        self.task = task
        self.imbalance_threshold = imbalance_threshold
        self.correlation_threshold = correlation_threshold

    def run(self) -> SanityReport:
        """Run all checks and return a SanityReport (never raises)."""
        logger.info("Running DatasetSanity checks (task=%s, target=%s)", self.task, self.target)

        # --- missing values ---
        try:
            check_missing_values(self.df)
            missing_result = CheckResult(passed=True, details={})
            logger.info("Missing values check: PASSED")
        except MissingValuesError as exc:
            missing_result = CheckResult(
                passed=False,
                details={"affected_columns": list(exc.columns) if exc.columns else []},
            )
            logger.warning("Missing values check: FAILED — %s", exc)
        except DatasetSanityError as exc:
            missing_result = CheckResult(passed=False, details={"error": str(exc)})
            logger.warning("Missing values check: FAILED — %s", exc)

        # --- class imbalance (classification only) ---
        if self.task == "classification":
            try:
                check_class_imbalance(self.df, target_column=self.target, threshold=self.imbalance_threshold)
                imbalance_result = CheckResult(passed=True, details={})
                logger.info("Class imbalance check: PASSED")
            except ClassImbalanceError as exc:
                imbalance_result = CheckResult(
                    passed=False,
                    details={
                        "target_column": exc.target_column,
                        "imbalance_ratio": exc.imbalance_ratio,
                    },
                )
                logger.warning("Class imbalance check: FAILED — %s", exc)
            except DatasetSanityError as exc:
                imbalance_result = CheckResult(passed=False, details={"error": str(exc)})
                logger.warning("Class imbalance check: FAILED — %s", exc)
        else:
            imbalance_result = CheckResult(passed=True, details={"skipped": "regression task"})

        # --- data leakage ---
        try:
            check_data_leakage(self.df, target_column=self.target, correlation_threshold=self.correlation_threshold)
            leakage_result = CheckResult(passed=True, details={})
            logger.info("Data leakage check: PASSED")
        except DataLeakageError as exc:
            leakage_result = CheckResult(
                passed=False,
                details={"leaked_features": list(exc.features) if exc.features else []},
            )
            logger.warning("Data leakage check: FAILED — %s", exc)
        except DatasetSanityError as exc:
            leakage_result = CheckResult(passed=False, details={"error": str(exc)})
            logger.warning("Data leakage check: FAILED — %s", exc)

        return SanityReport(
            missing_values=missing_result,
            class_imbalance=imbalance_result,
            leakage=leakage_result,
        )
