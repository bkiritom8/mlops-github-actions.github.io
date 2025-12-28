"""
Tests for data validation and quality
"""
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine


class TestDataQuality:
    """Test data quality and validation"""

    @pytest.fixture
    def sample_data(self):
        """Create sample wine dataset"""
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['quality'] = wine.target
        return df

    def test_data_shape(self, sample_data):
        """Test dataset has correct shape"""
        assert sample_data.shape[0] > 100, "Dataset should have more than 100 samples"
        assert sample_data.shape[1] > 10, "Dataset should have more than 10 features"

    def test_no_missing_values(self, sample_data):
        """Test dataset has no missing values"""
        missing = sample_data.isnull().sum().sum()
        assert missing == 0, f"Dataset should have no missing values, found {missing}"

    def test_numeric_types(self, sample_data):
        """Test all features are numeric"""
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) == len(sample_data.columns), \
            "All columns should be numeric"

    def test_no_infinite_values(self, sample_data):
        """Test dataset has no infinite values"""
        numeric_df = sample_data.select_dtypes(include=[np.number])
        infinite_values = np.isinf(numeric_df).any().any()
        assert not infinite_values, "Dataset should not contain infinite values"

    def test_target_distribution(self, sample_data):
        """Test target variable has reasonable distribution"""
        target_counts = sample_data['quality'].value_counts()

        # Each class should have at least 10 samples
        assert all(target_counts >= 10), \
            "Each class should have at least 10 samples"

    def test_feature_variance(self, sample_data):
        """Test features have non-zero variance"""
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            variance = sample_data[col].var()
            assert variance > 0, f"Feature {col} has zero variance"

    def test_data_ranges(self, sample_data):
        """Test data values are within reasonable ranges"""
        numeric_df = sample_data.select_dtypes(include=[np.number])

        # Check for extreme outliers (beyond 10 standard deviations)
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            min_val = numeric_df[col].min()
            max_val = numeric_df[col].max()

            # Values should be within 10 standard deviations
            assert min_val >= (mean - 10 * std), \
                f"Column {col} has extreme low outliers"
            assert max_val <= (mean + 10 * std), \
                f"Column {col} has extreme high outliers"


class TestDataFiles:
    """Test data file handling"""

    def test_data_directory_exists(self):
        """Test data directory exists"""
        assert os.path.exists('data'), "Data directory should exist"

    def test_can_load_wine_data(self):
        """Test wine dataset can be loaded"""
        wine = load_wine()
        assert wine.data.shape[0] > 0, "Wine dataset should have samples"
        assert wine.data.shape[1] > 0, "Wine dataset should have features"
