"""End-to-end training pipeline for MLOps demonstration."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


class TrainingPipeline:
    """End-to-end training pipeline."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "models/artifacts",
    ):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(str(self.output_dir))
        self.validator = DataValidator()
        self.trainer = ModelTrainer(str(self.output_dir))
        self.evaluator = ModelEvaluator(str(self.output_dir))

        # Pipeline state
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}
        self.artifacts = {}

    def run(
        self,
        dataset_type: str = "classification",
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        run_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Args:
            dataset_type: Type of dataset to use
            model_type: Type of model to train
            model_params: Optional model parameters
            run_validation: Whether to run data validation

        Returns:
            Pipeline results including metrics and artifact paths
        """
        print(f"\n{'='*60}")
        print(f"MLOps Training Pipeline - Run ID: {self.run_id}")
        print(f"{'='*60}\n")

        results = {
            "run_id": self.run_id,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Step 1: Load data
            print("Step 1: Loading data...")
            df, target = self.data_loader.load_demo_dataset(dataset_type)
            print(f"  Loaded {len(df)} samples with {len(df.columns)} features")
            results["data_info"] = {
                "n_samples": len(df),
                "n_features": len(df.columns),
                "dataset_type": dataset_type,
            }

            # Step 2: Validate data
            if run_validation:
                print("\nStep 2: Validating data...")
                validation_results = self.validator.run_all_validations(
                    df, target, list(df.columns)
                )
                all_valid = all(r.is_valid for r in validation_results.values())
                print(f"  Validation passed: {all_valid}")

                validation_report_path = self.output_dir / f"validation_{self.run_id}.json"
                self.validator.save_validation_report(
                    validation_results, str(validation_report_path)
                )
                results["validation"] = {
                    "passed": all_valid,
                    "report_path": str(validation_report_path),
                }
                self.artifacts["validation_report"] = str(validation_report_path)

            # Step 3: Preprocess data
            print("\nStep 3: Preprocessing data...")
            X, y = self.preprocessor.fit_transform(df, target)
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Save preprocessor
            preprocessor_path = self.preprocessor.save_preprocessors(f"preprocessor_{self.run_id}")
            self.artifacts["preprocessor"] = str(preprocessor_path)

            # Step 4: Train model
            print(f"\nStep 4: Training {model_type} model...")
            self.trainer.create_model(model_type, model_params)
            training_metrics = self.trainer.train(X_train, y_train, X_val, y_val)
            print(f"  Training accuracy: {training_metrics['train_accuracy']:.4f}")
            print(f"  Validation accuracy: {training_metrics['val_accuracy']:.4f}")
            print(
                f"  CV mean: {training_metrics['cv_mean']:.4f} (+/- {training_metrics['cv_std']:.4f})"
            )
            results["training_metrics"] = training_metrics

            # Step 5: Evaluate on test set
            print("\nStep 5: Evaluating model...")
            y_pred = self.trainer.predict(X_test)
            y_proba = self.trainer.predict_proba(X_test)

            eval_metrics = self.evaluator.evaluate(y_test, y_pred, y_proba)
            print(f"  Test accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Test F1 score: {eval_metrics['f1_score']:.4f}")
            if eval_metrics.get("roc_auc"):
                print(f"  Test ROC AUC: {eval_metrics['roc_auc']:.4f}")
            results["evaluation_metrics"] = eval_metrics

            # Save evaluation
            eval_path = self.evaluator.save_evaluation(eval_metrics, f"evaluation_{self.run_id}")
            self.artifacts["evaluation"] = str(eval_path)

            # Step 6: Get feature importance
            print("\nStep 6: Analyzing feature importance...")
            feature_importance = self.trainer.get_feature_importance(
                self.preprocessor.get_feature_importance_names()
            )
            top_features = list(feature_importance.items())[:5]
            print("  Top 5 features:")
            for feat, imp in top_features:
                print(f"    {feat}: {imp:.4f}")
            results["feature_importance"] = feature_importance

            # Step 7: Save model
            print("\nStep 7: Saving model artifacts...")
            model_path, metadata_path = self.trainer.save_model(f"model_{self.run_id}")
            print(f"  Model saved to: {model_path}")
            self.artifacts["model"] = str(model_path)
            self.artifacts["model_metadata"] = str(metadata_path)

            # Generate final report
            report = self.evaluator.generate_report(
                eval_metrics, model_name=f"{model_type}_{self.run_id}"
            )
            report_path = self.output_dir / f"report_{self.run_id}.txt"
            with open(report_path, "w") as f:
                f.write(report)
            self.artifacts["report"] = str(report_path)

            # Final status
            results["status"] = "completed"
            results["artifacts"] = self.artifacts

            print(f"\n{'='*60}")
            print("Pipeline completed successfully!")
            print(f"{'='*60}\n")

            # Save pipeline results
            results_path = self.output_dir / f"pipeline_results_{self.run_id}.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"\nPipeline failed: {e}")
            raise

    def generate_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Generate metrics in format suitable for dashboard display."""
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
