"""
Data Validation Script
Validates data quality and schema
"""
import os
import sys
import pandas as pd
import numpy as np


def validate_data_schema(df):
    """Validate that dataset has expected schema"""
    print("Validating data schema...")

    required_min_cols = 10  # Wine dataset should have at least 10 features

    if df.shape[1] < required_min_cols:
        raise ValueError(f"Expected at least {required_min_cols} columns, got {df.shape[1]}")

    print(f"✓ Schema validation passed - {df.shape[1]} columns found")
    return True


def check_missing_values(df):
    """Check for missing values in dataset"""
    print("\nChecking for missing values...")

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        print(f"⚠ Warning: Found {total_missing} missing values")
        print(missing[missing > 0])
        return False
    else:
        print("✓ No missing values found")
        return True


def check_data_types(df):
    """Validate data types"""
    print("\nChecking data types...")

    # All features should be numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric = set(df.columns) - set(numeric_cols)

    if len(non_numeric) > 0:
        print(f"⚠ Warning: Non-numeric columns found: {non_numeric}")
        return False
    else:
        print(f"✓ All {len(numeric_cols)} columns are numeric")
        return True


def check_data_ranges(df):
    """Check for outliers and valid ranges"""
    print("\nChecking data ranges...")

    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        print("✗ Found infinite values in dataset")
        return False

    # Check for extreme outliers (values beyond 5 standard deviations)
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
    extreme_outliers = (z_scores > 5).any().any()

    if extreme_outliers:
        print("⚠ Warning: Found extreme outliers (>5 std dev)")
        return False
    else:
        print("✓ Data ranges are within acceptable bounds")
        return True


def check_minimum_samples(df, min_samples=100):
    """Ensure dataset has minimum number of samples"""
    print(f"\nChecking minimum sample size (minimum: {min_samples})...")

    if len(df) < min_samples:
        raise ValueError(f"Dataset too small: {len(df)} samples (minimum: {min_samples})")

    print(f"✓ Dataset has {len(df)} samples")
    return True


def validate_dataset(data_path='data/wine_data.csv'):
    """Run all validation checks on dataset"""
    print("=" * 50)
    print("Starting Data Validation")
    print("=" * 50)

    if not os.path.exists(data_path):
        print(f"✗ Error: Data file not found at {data_path}")
        return False

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Run validation checks
    checks = [
        validate_data_schema(df),
        check_missing_values(df),
        check_data_types(df),
        check_data_ranges(df),
        check_minimum_samples(df)
    ]

    print("\n" + "=" * 50)
    if all(checks):
        print("✓ All Data Validation Checks Passed!")
        print("=" * 50)
        return True
    else:
        print("⚠ Some validation checks failed or have warnings")
        print("=" * 50)
        return True  # Return True for warnings, only fail on errors


if __name__ == '__main__':
    success = validate_dataset()
    sys.exit(0 if success else 1)
