"""Data validation utilities for MLOps pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ValidationResult:
    """Container for validation results."""

    is_valid: bool
    checks_passed: int
    checks_failed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataValidator:
    """Validate data quality and schema for ML pipeline."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.validation_history = []

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        expected_dtypes: Optional[Dict[str, str]] = None,
    ) -> ValidationResult:
        """
        Validate DataFrame schema.

        Args:
            df: Input DataFrame
            expected_columns: List of expected column names
            expected_dtypes: Optional dict of column name to expected dtype

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        details = {}

        # Check columns
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)

        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            warnings.append(f"Extra columns found: {extra_cols}")

        details["missing_columns"] = list(missing_cols)
        details["extra_columns"] = list(extra_cols)

        # Check dtypes if provided
        if expected_dtypes:
            dtype_mismatches = {}
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if not self._dtype_compatible(actual_dtype, expected_dtype):
                        dtype_mismatches[col] = {
                            "expected": expected_dtype,
                            "actual": actual_dtype,
                        }
            if dtype_mismatches:
                warnings.append(f"Dtype mismatches: {dtype_mismatches}")
            details["dtype_mismatches"] = dtype_mismatches

        is_valid = len(errors) == 0
        checks_passed = 2 - len(errors)
        checks_failed = len(errors)

        return ValidationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        max_missing_ratio: float = 0.1,
        check_duplicates: bool = True,
        numeric_bounds: Optional[Dict[str, tuple]] = None,
    ) -> ValidationResult:
        """
        Validate data quality.

        Args:
            df: Input DataFrame
            max_missing_ratio: Maximum allowed ratio of missing values
            check_duplicates: Whether to check for duplicate rows
            numeric_bounds: Optional dict of column name to (min, max) bounds

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        details = {}

        # Check missing values
        missing_ratios = df.isnull().mean()
        high_missing = missing_ratios[missing_ratios > max_missing_ratio]
        if len(high_missing) > 0:
            errors.append(
                f"Columns with high missing ratio (>{max_missing_ratio}): "
                f"{high_missing.to_dict()}"
            )
        details["missing_ratios"] = missing_ratios.to_dict()

        # Check duplicates
        if check_duplicates:
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                warnings.append(f"Found {n_duplicates} duplicate rows")
            details["n_duplicates"] = int(n_duplicates)

        # Check numeric bounds
        if numeric_bounds:
            bound_violations = {}
            for col, (min_val, max_val) in numeric_bounds.items():
                if col in df.columns:
                    below_min = (df[col] < min_val).sum()
                    above_max = (df[col] > max_val).sum()
                    if below_min > 0 or above_max > 0:
                        bound_violations[col] = {
                            "below_min": int(below_min),
                            "above_max": int(above_max),
                        }
            if bound_violations:
                warnings.append(f"Bound violations: {bound_violations}")
            details["bound_violations"] = bound_violations

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                inf_counts[col] = int(n_inf)
        if inf_counts:
            errors.append(f"Infinite values found: {inf_counts}")
        details["infinite_values"] = inf_counts

        is_valid = len(errors) == 0
        checks_passed = 4 - len(errors)
        checks_failed = len(errors)

        return ValidationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def validate_target_distribution(
        self,
        target: pd.Series,
        min_class_ratio: float = 0.1,
    ) -> ValidationResult:
        """
        Validate target variable distribution.

        Args:
            target: Target Series
            min_class_ratio: Minimum ratio for any class

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        details = {}

        # Class distribution
        class_counts = target.value_counts()
        class_ratios = target.value_counts(normalize=True)
        details["class_counts"] = class_counts.to_dict()
        details["class_ratios"] = class_ratios.to_dict()

        # Check for class imbalance
        min_ratio = class_ratios.min()
        if min_ratio < min_class_ratio:
            warnings.append(f"Class imbalance detected. Minimum class ratio: {min_ratio:.3f}")
            details["is_imbalanced"] = True
        else:
            details["is_imbalanced"] = False

        # Check for missing targets
        n_missing = target.isnull().sum()
        if n_missing > 0:
            errors.append(f"Found {n_missing} missing target values")
        details["n_missing_targets"] = int(n_missing)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            checks_passed=2 - len(errors),
            checks_failed=len(errors),
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def run_all_validations(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        expected_columns: Optional[List[str]] = None,
    ) -> Dict[str, ValidationResult]:
        """
        Run all validation checks.

        Args:
            df: Input DataFrame
            target: Target Series
            expected_columns: Optional list of expected columns

        Returns:
            Dict of validation results by check name
        """
        results = {}

        # Schema validation
        if expected_columns:
            results["schema"] = self.validate_schema(df, expected_columns)
        else:
            results["schema"] = self.validate_schema(df, list(df.columns))

        # Data quality validation
        results["data_quality"] = self.validate_data_quality(df)

        # Target validation
        results["target_distribution"] = self.validate_target_distribution(target)

        # Store in history
        self.validation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "results": {k: v.__dict__ for k, v in results.items()},
            }
        )

        return results

    def save_validation_report(
        self, results: Dict[str, ValidationResult], output_path: str
    ) -> Path:
        """Save validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "all_valid": all(r.is_valid for r in results.values()),
                "total_checks_passed": sum(r.checks_passed for r in results.values()),
                "total_checks_failed": sum(r.checks_failed for r in results.values()),
            },
            "details": {k: v.__dict__ for k, v in results.items()},
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected."""
        dtype_groups = {
            "numeric": ["int64", "int32", "float64", "float32", "int", "float"],
            "string": ["object", "string", "str"],
            "datetime": ["datetime64", "datetime"],
            "boolean": ["bool", "boolean"],
        }

        for group_types in dtype_groups.values():
            if actual in group_types and expected in group_types:
                return True

        return actual == expected
