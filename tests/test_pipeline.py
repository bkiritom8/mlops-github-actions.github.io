"""Tests for pipeline modules."""

import tempfile
import json
from pathlib import Path

from src.pipelines.training_pipeline import TrainingPipeline


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)

            assert pipeline.data_loader is not None
            assert pipeline.preprocessor is not None
            assert pipeline.validator is not None
            assert pipeline.trainer is not None
            assert pipeline.evaluator is not None

    def test_pipeline_run_classification(self):
        """Test running pipeline with classification dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)
            results = pipeline.run(
                dataset_type="classification",
                model_type="random_forest",
            )

            assert results["status"] == "completed"
            assert "run_id" in results
            assert "training_metrics" in results
            assert "evaluation_metrics" in results
            assert results["evaluation_metrics"]["accuracy"] > 0

    def test_pipeline_run_iris(self):
        """Test running pipeline with iris dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)
            results = pipeline.run(
                dataset_type="iris",
                model_type="random_forest",
            )

            assert results["status"] == "completed"

    def test_pipeline_run_without_validation(self):
        """Test running pipeline without validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)
            results = pipeline.run(
                dataset_type="classification",
                model_type="random_forest",
                run_validation=False,
            )

            assert results["status"] == "completed"
            assert "validation" not in results

    def test_pipeline_saves_artifacts(self):
        """Test that pipeline saves all artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)
            results = pipeline.run(
                dataset_type="classification",
                model_type="random_forest",
            )

            assert "artifacts" in results
            assert "model" in results["artifacts"]

            # Check model file exists
            model_path = Path(results["artifacts"]["model"])
            assert model_path.exists()

    def test_pipeline_different_models(self):
        """Test pipeline with different model types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for model_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
                pipeline = TrainingPipeline(output_dir=tmpdir)
                results = pipeline.run(
                    dataset_type="classification",
                    model_type=model_type,
                    run_validation=False,
                )

                assert results["status"] == "completed", f"Failed for {model_type}"

    def test_pipeline_results_saved_to_file(self):
        """Test that pipeline results are saved to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = TrainingPipeline(output_dir=tmpdir)
            results = pipeline.run(
                dataset_type="classification",
                model_type="random_forest",
            )

            # Check results file exists
            results_file = Path(tmpdir) / f"pipeline_results_{results['run_id']}.json"
            assert results_file.exists()

            # Verify contents
            with open(results_file) as f:
                saved_results = json.load(f)

            assert saved_results["status"] == "completed"
