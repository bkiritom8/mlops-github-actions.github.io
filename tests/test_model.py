"""
Tests for model training and evaluation
"""
import os
import json
import pytest
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import prepare_data, train_model, evaluate_model


class TestModelTraining:
    """Test model training pipeline"""

    def test_data_preparation(self):
        """Test data preparation function"""
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            test_size=0.2, random_state=42
        )

        # Check shapes
        assert X_train.shape[0] > 0, "Training set should not be empty"
        assert X_test.shape[0] > 0, "Test set should not be empty"
        assert len(y_train) == X_train.shape[0], "y_train length mismatch"
        assert len(y_test) == X_test.shape[0], "y_test length mismatch"

        # Check scaling
        assert np.abs(X_train.mean()) < 1.0, "Data should be scaled"
        assert scaler is not None, "Scaler should be returned"

    def test_model_training(self):
        """Test model training function"""
        X_train, _, y_train, _, _ = prepare_data(random_state=42)

        model = train_model(
            X_train, y_train,
            n_estimators=10,  # Small for testing
            max_depth=5,
            random_state=42
        )

        assert model is not None, "Model should be created"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'feature_importances_'), "Model should have feature importances"

    def test_model_evaluation(self):
        """Test model evaluation function"""
        X_train, X_test, y_train, y_test, _ = prepare_data(random_state=42)

        model = train_model(X_train, y_train, n_estimators=10, random_state=42)
        metrics = evaluate_model(model, X_test, y_test)

        # Check all metrics are present
        assert 'accuracy' in metrics, "Accuracy should be in metrics"
        assert 'precision' in metrics, "Precision should be in metrics"
        assert 'recall' in metrics, "Recall should be in metrics"
        assert 'f1_score' in metrics, "F1-score should be in metrics"

        # Check metrics are in valid range
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"
        assert 0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1"
        assert 0 <= metrics['f1_score'] <= 1, "F1-score should be between 0 and 1"


class TestModelQuality:
    """Test model quality and performance"""

    def test_model_accuracy_threshold(self):
        """Test that model meets minimum accuracy threshold"""
        # Check if metrics file exists
        metrics_path = 'reports/metrics.json'

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            min_accuracy = 0.80  # Minimum acceptable accuracy
            assert metrics['accuracy'] >= min_accuracy, \
                f"Model accuracy {metrics['accuracy']:.4f} below threshold {min_accuracy}"

    def test_saved_model_exists(self):
        """Test that model artifacts are saved"""
        if os.path.exists('models/model.joblib'):
            model = joblib.load('models/model.joblib')
            assert model is not None, "Saved model should load correctly"

    def test_model_predictions(self):
        """Test model can make predictions"""
        if os.path.exists('models/model.joblib') and os.path.exists('models/scaler.joblib'):
            model = joblib.load('models/model.joblib')
            scaler = joblib.load('models/scaler.joblib')

            # Load test data
            wine = load_wine()
            X_test = wine.data[:5]  # Test with 5 samples
            X_test_scaled = scaler.transform(X_test)

            predictions = model.predict(X_test_scaled)

            assert len(predictions) == 5, "Should predict for all samples"
            assert all(pred in [0, 1, 2] for pred in predictions), \
                "Predictions should be valid class labels"
