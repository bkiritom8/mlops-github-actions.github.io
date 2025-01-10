"""Tests for data processing modules."""

import pytest
import pandas as pd
import numpy as np
import tempfile

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_demo_dataset_classification(self):
        """Test loading classification dataset."""
        loader = DataLoader()
        df, target = loader.load_demo_dataset("classification")

        assert isinstance(df, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(df) == 1000
        assert len(target) == 1000
        assert df.shape[1] == 20

    def test_load_demo_dataset_iris(self):
        """Test loading iris dataset."""
        loader = DataLoader()
        df, target = loader.load_demo_dataset("iris")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert len(target.unique()) == 3

    def test_load_demo_dataset_wine(self):
        """Test loading wine dataset."""
        loader = DataLoader()
        df, target = loader.load_demo_dataset("wine")

        assert isinstance(df, pd.DataFrame)
        assert len(target.unique()) == 3

    def test_invalid_dataset_type(self):
        """Test error on invalid dataset type."""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load_demo_dataset("invalid")

    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            df, target = loader.load_demo_dataset("classification")

            # Save
            loader.save_dataset(df, target, "test_data")

            # Load
            df_loaded, target_loaded = loader.load_dataset("test_data")

            assert len(df_loaded) == len(df)
            assert len(target_loaded) == len(target)

    def test_get_data_statistics(self):
        """Test data statistics calculation."""
        loader = DataLoader()
        df, _ = loader.load_demo_dataset("classification")

        stats = loader.get_data_statistics(df)

        assert "n_samples" in stats
        assert "n_features" in stats
        assert "numeric_stats" in stats
        assert stats["n_samples"] == 1000


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        target = pd.Series(np.random.randint(0, 2, 100))
        return df, target

    def test_fit_transform_standard(self, sample_data):
        """Test fit_transform with standard scaling."""
        df, target = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = DataPreprocessor(tmpdir)
            X, y = preprocessor.fit_transform(df, target, scaling="standard")

            assert X.shape == (100, 3)
            assert len(y) == 100
            # Standard scaled data should have mean near 0
            assert abs(X.mean()) < 0.1

    def test_fit_transform_minmax(self, sample_data):
        """Test fit_transform with minmax scaling."""
        df, target = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = DataPreprocessor(tmpdir)
            X, y = preprocessor.fit_transform(df, target, scaling="minmax")

            # MinMax scaled data should be between 0 and 1
            assert X.min() >= 0
            assert X.max() <= 1

    def test_split_data(self, sample_data):
        """Test data splitting."""
        df, target = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = DataPreprocessor(tmpdir)
            X, y = preprocessor.fit_transform(df, target)

            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
                X, y, train_size=0.8, val_size=0.1, test_size=0.1
            )

            total = len(X_train) + len(X_val) + len(X_test)
            assert total == 100

    def test_save_and_load_preprocessors(self, sample_data):
        """Test saving and loading preprocessors."""
        df, target = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = DataPreprocessor(tmpdir)
            preprocessor.fit_transform(df, target)
            preprocessor.save_preprocessors("test_preprocessor")

            # Load in new instance
            preprocessor2 = DataPreprocessor(tmpdir)
            preprocessor2.load_preprocessors("test_preprocessor")

            assert preprocessor2.scaler is not None


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [1.0, 2.0, None, 4.0, 5.0],
                "feature_3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        target = pd.Series([0, 1, 0, 1, 0])
        return df, target

    def test_validate_schema_pass(self, sample_data):
        """Test schema validation pass."""
        df, _ = sample_data
        validator = DataValidator()

        result = validator.validate_schema(
            df, expected_columns=["feature_1", "feature_2", "feature_3"]
        )

        assert result.is_valid

    def test_validate_schema_missing_columns(self, sample_data):
        """Test schema validation with missing columns."""
        df, _ = sample_data
        validator = DataValidator()

        result = validator.validate_schema(
            df, expected_columns=["feature_1", "feature_2", "feature_3", "feature_4"]
        )

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_data_quality(self, sample_data):
        """Test data quality validation."""
        df, _ = sample_data
        validator = DataValidator()

        result = validator.validate_data_quality(df, max_missing_ratio=0.5)

        assert result.is_valid

    def test_validate_target_distribution(self, sample_data):
        """Test target distribution validation."""
        _, target = sample_data
        validator = DataValidator()

        result = validator.validate_target_distribution(target)

        assert result.is_valid
        assert "class_counts" in result.details

    def test_run_all_validations(self, sample_data):
        """Test running all validations."""
        df, target = sample_data
        validator = DataValidator()

        results = validator.run_all_validations(df, target)

        assert "schema" in results
        assert "data_quality" in results
        assert "target_distribution" in results
