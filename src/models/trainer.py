"""Model training utilities for MLOps pipeline."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
from pathlib import Path
from datetime import datetime
import json


class ModelTrainer:
    """Train and manage ML models."""

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
    }

    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        },
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        },
        "svm": {
            "kernel": "rbf",
            "probability": True,
            "random_state": 42,
        },
    }

    def __init__(self, artifacts_path: str = "models/artifacts"):
        self.artifacts_path = Path(artifacts_path)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_type = None
        self.training_metadata = {}

    def create_model(
        self,
        model_type: str = "random_forest",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new model instance.

        Args:
            model_type: Type of model to create
            params: Optional model parameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        model_class = self.SUPPORTED_MODELS[model_type]
        model_params = self.DEFAULT_PARAMS.get(model_type, {}).copy()

        if params:
            model_params.update(params)

        self.model = model_class(**model_params)
        self.model_type = model_type
        self.training_metadata["model_type"] = model_type
        self.training_metadata["model_params"] = model_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        start_time = datetime.now()

        # Fit the model
        self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Calculate training metrics
        train_score = self.model.score(X_train, y_train)
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)

        metrics = {
            "train_accuracy": float(train_score),
            "val_accuracy": float(val_score) if val_score else None,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "training_time_seconds": training_time,
            "n_train_samples": len(X_train),
            "n_features": X_train.shape[1],
        }

        self.training_metadata.update(
            {
                "training_metrics": metrics,
                "training_timestamp": datetime.now().isoformat(),
            }
        )

        return metrics

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, list],
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Best parameters and scores
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
            },
        }

        self.training_metadata["hyperparameter_search"] = results

        return results

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dict of feature name to importance score
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            return {}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save_model(self, name: str = "model") -> Tuple[Path, Path]:
        """
        Save model and metadata to disk.

        Args:
            name: Model name

        Returns:
            Tuple of (model_path, metadata_path)
        """
        if self.model is None:
            raise ValueError("No model to save.")

        model_path = self.artifacts_path / f"{name}.joblib"
        joblib.dump(self.model, model_path)

        metadata_path = self.artifacts_path / f"{name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)

        return model_path, metadata_path

    def load_model(self, name: str = "model") -> None:
        """
        Load model and metadata from disk.

        Args:
            name: Model name
        """
        model_path = self.artifacts_path / f"{name}.joblib"
        self.model = joblib.load(model_path)

        metadata_path = self.artifacts_path / f"{name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.training_metadata = json.load(f)
            self.model_type = self.training_metadata.get("model_type")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise ValueError("Model does not support probability predictions.")
