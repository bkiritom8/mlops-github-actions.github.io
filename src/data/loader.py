"""Data loading utilities for MLOps pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.datasets import make_classification, load_iris, load_wine
import json


class DataLoader:
    """Load and manage datasets for ML pipeline."""

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_demo_dataset(
        self, dataset_type: str = "classification"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate or load a demo dataset for demonstration.

        Args:
            dataset_type: Type of dataset ('classification', 'iris', 'wine')

        Returns:
            Tuple of features DataFrame and target Series
        """
        if dataset_type == "classification":
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42,
                flip_y=0.1,
            )
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            target = pd.Series(y, name="target")

        elif dataset_type == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            target = pd.Series(data.target, name="target")

        elif dataset_type == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            target = pd.Series(data.target, name="target")

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return df, target

    def save_dataset(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        name: str,
        location: str = "processed",
    ) -> Path:
        """
        Save dataset to disk.

        Args:
            df: Features DataFrame
            target: Target Series
            name: Dataset name
            location: 'raw' or 'processed'

        Returns:
            Path to saved dataset
        """
        save_path = self.raw_path if location == "raw" else self.processed_path
        full_df = df.copy()
        full_df["target"] = target

        file_path = save_path / f"{name}.csv"
        full_df.to_csv(file_path, index=False)

        # Save metadata
        metadata = {
            "name": name,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "feature_names": list(df.columns),
            "target_name": "target",
            "target_classes": list(target.unique()),
        }
        metadata_path = save_path / f"{name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return file_path

    def load_dataset(
        self, name: str, location: str = "processed"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from disk.

        Args:
            name: Dataset name
            location: 'raw' or 'processed'

        Returns:
            Tuple of features DataFrame and target Series
        """
        load_path = self.raw_path if location == "raw" else self.processed_path
        file_path = load_path / f"{name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path)
        target = df.pop("target")

        return df, target

    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate statistics for a dataset."""
        stats = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_stats": {},
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

        return stats
