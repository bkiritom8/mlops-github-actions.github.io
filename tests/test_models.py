"""Tests for model training and evaluation modules."""

import pytest
import numpy as np
import tempfile

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        return X_train, y_train, X_val, y_val

    def test_create_model_random_forest(self):
        """Test creating random forest model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("random_forest")

            assert trainer.model is not None
            assert trainer.model_type == "random_forest"

    def test_create_model_gradient_boosting(self):
        """Test creating gradient boosting model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("gradient_boosting")

            assert trainer.model is not None
            assert trainer.model_type == "gradient_boosting"

    def test_create_model_invalid_type(self):
        """Test error on invalid model type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)

            with pytest.raises(ValueError):
                trainer.create_model("invalid_model")

    def test_train_model(self, sample_data):
        """Test training a model."""
        X_train, y_train, X_val, y_val = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("random_forest")

            metrics = trainer.train(X_train, y_train, X_val, y_val)

            assert "train_accuracy" in metrics
            assert "val_accuracy" in metrics
            assert "cv_mean" in metrics
            assert metrics["train_accuracy"] > 0

    def test_predict(self, sample_data):
        """Test making predictions."""
        X_train, y_train, X_val, y_val = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("random_forest")
            trainer.train(X_train, y_train)

            predictions = trainer.predict(X_val)

            assert len(predictions) == len(X_val)
            assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_data):
        """Test getting prediction probabilities."""
        X_train, y_train, X_val, y_val = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("random_forest")
            trainer.train(X_train, y_train)

            probabilities = trainer.predict_proba(X_val)

            assert probabilities.shape == (len(X_val), 2)
            assert all(0 <= p <= 1 for p in probabilities.flatten())

    def test_get_feature_importance(self, sample_data):
        """Test getting feature importance."""
        X_train, y_train, _, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(tmpdir)
            trainer.create_model("random_forest")
            trainer.train(X_train, y_train)

            importance = trainer.get_feature_importance()

            assert len(importance) == 10

    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        X_train, y_train, X_val, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            trainer1 = ModelTrainer(tmpdir)
            trainer1.create_model("random_forest")
            trainer1.train(X_train, y_train)
            trainer1.save_model("test_model")

            # Load in new trainer
            trainer2 = ModelTrainer(tmpdir)
            trainer2.load_model("test_model")

            # Predictions should match
            pred1 = trainer1.predict(X_val)
            pred2 = trainer2.predict(X_val)

            np.testing.assert_array_equal(pred1, pred2)


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
        y_proba = np.random.rand(10, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_proba

    def test_evaluate_basic(self, sample_predictions):
        """Test basic evaluation."""
        y_true, y_pred, _ = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            metrics = evaluator.evaluate(y_true, y_pred)

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "confusion_matrix" in metrics

    def test_evaluate_with_probabilities(self, sample_predictions):
        """Test evaluation with probabilities."""
        y_true, y_pred, y_proba = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            metrics = evaluator.evaluate(y_true, y_pred, y_proba)

            assert "roc_auc" in metrics

    def test_get_roc_curve_data(self, sample_predictions):
        """Test ROC curve data generation."""
        y_true, _, y_proba = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            roc_data = evaluator.get_roc_curve_data(y_true, y_proba)

            assert "fpr" in roc_data
            assert "tpr" in roc_data
            assert "thresholds" in roc_data

    def test_generate_report(self, sample_predictions):
        """Test report generation."""
        y_true, y_pred, _ = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            metrics = evaluator.evaluate(y_true, y_pred)
            report = evaluator.generate_report(metrics, "Test Model")

            assert "Model Evaluation Report" in report
            assert "Accuracy" in report

    def test_check_performance_threshold(self, sample_predictions):
        """Test performance threshold checking."""
        y_true, y_pred, _ = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            metrics = evaluator.evaluate(y_true, y_pred)

            thresholds = {"accuracy": 0.5, "f1_score": 0.5}
            results = evaluator.check_performance_threshold(metrics, thresholds)

            assert "accuracy" in results
            assert "f1_score" in results

    def test_save_evaluation(self, sample_predictions):
        """Test saving evaluation results."""
        y_true, y_pred, _ = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(tmpdir)
            metrics = evaluator.evaluate(y_true, y_pred)

            output_path = evaluator.save_evaluation(metrics, "test_eval")

            assert output_path.exists()
