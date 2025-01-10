"""Inference pipeline for model serving."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json

from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer


class InferencePipeline:
    """Production inference pipeline."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        artifacts_dir: str = "models/artifacts",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        # Load model and preprocessor
        self.trainer = ModelTrainer(str(self.artifacts_dir))
        self.preprocessor = DataPreprocessor(str(self.artifacts_dir))

        self._load_artifacts()

        # Inference statistics
        self.inference_count = 0
        self.inference_history = []

    def _load_artifacts(self) -> None:
        """Load model and preprocessor from disk."""
        # Extract name from path for loading
        model_name = Path(self.model_path).stem
        self.trainer.load_model(model_name)

        preprocessor_name = Path(self.preprocessor_path).stem
        self.preprocessor.load_preprocessors(preprocessor_name)

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        return_proba: bool = False,
    ) -> Dict[str, Any]:
        """
        Make predictions on new data.

        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_proba: Whether to return prediction probabilities

        Returns:
            Prediction results
        """
        start_time = datetime.now()

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Ensure correct column order
        expected_features = self.preprocessor.get_feature_importance_names()
        if expected_features:
            # Handle missing columns
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_features]

        # Preprocess
        X = self.preprocessor.transform(df)

        # Predict
        predictions = self.trainer.predict(X)

        result = {
            "predictions": predictions.tolist(),
            "n_samples": len(predictions),
            "timestamp": datetime.now().isoformat(),
        }

        if return_proba:
            try:
                probabilities = self.trainer.predict_proba(X)
                result["probabilities"] = probabilities.tolist()
            except ValueError:
                result["probabilities"] = None

        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        result["inference_time_ms"] = inference_time

        # Update statistics
        self.inference_count += len(predictions)
        self.inference_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(predictions),
                "inference_time_ms": inference_time,
            }
        )

        return result

    def predict_single(
        self, features: Dict[str, Any], return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample.

        Args:
            features: Feature dictionary
            return_proba: Whether to return probabilities

        Returns:
            Single prediction result
        """
        result = self.predict(features, return_proba)
        return {
            "prediction": result["predictions"][0],
            "probability": (result["probabilities"][0] if result.get("probabilities") else None),
            "inference_time_ms": result["inference_time_ms"],
        }

    def batch_predict(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        return_proba: bool = False,
    ) -> Dict[str, Any]:
        """
        Make predictions in batches.

        Args:
            data: List of feature dictionaries
            batch_size: Size of each batch
            return_proba: Whether to return probabilities

        Returns:
            Batch prediction results
        """
        all_predictions = []
        all_probabilities = []
        total_time = 0

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            result = self.predict(batch, return_proba)
            all_predictions.extend(result["predictions"])
            if result.get("probabilities"):
                all_probabilities.extend(result["probabilities"])
            total_time += result["inference_time_ms"]

        return {
            "predictions": all_predictions,
            "probabilities": all_probabilities if all_probabilities else None,
            "n_samples": len(all_predictions),
            "total_inference_time_ms": total_time,
            "avg_inference_time_ms": total_time / len(data) if data else 0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        if not self.inference_history:
            return {
                "total_predictions": 0,
                "avg_inference_time_ms": 0,
            }

        times = [h["inference_time_ms"] for h in self.inference_history]
        return {
            "total_predictions": self.inference_count,
            "total_requests": len(self.inference_history),
            "avg_inference_time_ms": np.mean(times),
            "min_inference_time_ms": np.min(times),
            "max_inference_time_ms": np.max(times),
            "p95_inference_time_ms": np.percentile(times, 95),
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Try a dummy prediction
            n_features = len(self.preprocessor.get_feature_importance_names())
            dummy_data = {f"feature_{i}": 0.0 for i in range(n_features)}
            self.predict_single(dummy_data)

            return {
                "status": "healthy",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
