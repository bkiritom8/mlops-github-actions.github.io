"""Data preprocessing utilities for MLOps pipeline."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


class DataPreprocessor:
    """Preprocess data for ML training and inference."""

    def __init__(self, artifacts_path: str = "models/artifacts"):
        self.artifacts_path = Path(artifacts_path)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_names = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        scaling: str = "standard",
        handle_missing: str = "mean",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessors and transform data.

        Args:
            df: Input DataFrame
            target: Optional target Series
            scaling: Scaling method ('standard', 'minmax', 'none')
            handle_missing: Missing value strategy ('mean', 'median', 'most_frequent', 'drop')

        Returns:
            Transformed features and optionally transformed target
        """
        self.feature_names = list(df.columns)

        # Handle missing values
        if handle_missing != "drop":
            self.imputer = SimpleImputer(strategy=handle_missing)
            X = self.imputer.fit_transform(df)
        else:
            df = df.dropna()
            X = df.values
            if target is not None:
                target = target.loc[df.index]

        # Apply scaling
        if scaling == "standard":
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)

        y = target.values if target is not None else None

        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.

        Args:
            df: Input DataFrame

        Returns:
            Transformed features
        """
        X = df.values

        if self.imputer is not None:
            X = self.imputer.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Features array
            y: Target array
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: separate validation from training
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_preprocessors(self, name: str = "preprocessor") -> Path:
        """Save fitted preprocessors to disk."""
        preprocessor_data = {
            "scaler": self.scaler,
            "imputer": self.imputer,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
        }
        save_path = self.artifacts_path / f"{name}.joblib"
        joblib.dump(preprocessor_data, save_path)
        return save_path

    def load_preprocessors(self, name: str = "preprocessor") -> None:
        """Load preprocessors from disk."""
        load_path = self.artifacts_path / f"{name}.joblib"
        preprocessor_data = joblib.load(load_path)
        self.scaler = preprocessor_data["scaler"]
        self.imputer = preprocessor_data["imputer"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.feature_names = preprocessor_data["feature_names"]

    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis."""
        return self.feature_names or []
