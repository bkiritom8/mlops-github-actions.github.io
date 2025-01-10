"""Model evaluation utilities for MLOps pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from datetime import datetime
import json
from pathlib import Path


class ModelEvaluator:
    """Evaluate ML model performance."""

    def __init__(self, output_path: str = "models/artifacts"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.evaluation_history = []

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Optional prediction probabilities
            labels: Optional class labels

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_true),
        }

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Handle binary vs multiclass
        is_binary = len(np.unique(y_true)) == 2
        average = "binary" if is_binary else "weighted"

        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

        # ROC AUC (only for binary or if probabilities provided)
        if y_proba is not None:
            try:
                if is_binary:
                    if y_proba.ndim == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            except ValueError:
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        # Calculate additional metrics for binary classification
        if is_binary:
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        self.evaluation_history.append(metrics)

        return metrics

    def get_roc_curve_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, List[float]]:
        """
        Get ROC curve data points.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    def get_precision_recall_curve_data(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Get Precision-Recall curve data points.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary with precision, recall, and thresholds
        """
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

    def compare_models(self, evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model evaluations.

        Args:
            evaluations: Dict of model name to evaluation metrics

        Returns:
            DataFrame comparing models
        """
        comparison_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        data = []

        for model_name, metrics in evaluations.items():
            row = {"model": model_name}
            for metric in comparison_metrics:
                row[metric] = metrics.get(metric)
            data.append(row)

        return pd.DataFrame(data).set_index("model")

    def save_evaluation(self, metrics: Dict[str, Any], name: str = "evaluation") -> Path:
        """
        Save evaluation metrics to JSON file.

        Args:
            metrics: Evaluation metrics dictionary
            name: Output file name

        Returns:
            Path to saved file
        """
        output_file = self.output_path / f"{name}.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return output_file

    def generate_report(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        include_curves: bool = True,
        y_true: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ) -> str:
        """
        Generate a text report of evaluation results.

        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            include_curves: Whether to include curve data
            y_true: True labels (needed for curves)
            y_proba: Prediction probabilities (needed for curves)

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            f"Model Evaluation Report: {model_name}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "PERFORMANCE METRICS",
            "-" * 40,
            f"Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}",
            f"Precision: {metrics.get('precision', 'N/A'):.4f}",
            f"Recall:    {metrics.get('recall', 'N/A'):.4f}",
            f"F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}",
        ]

        if metrics.get("roc_auc") is not None:
            report_lines.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        report_lines.extend(
            [
                "",
                "CONFUSION MATRIX",
                "-" * 40,
            ]
        )

        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            report_lines.append(str(cm))

        if metrics.get("true_positives") is not None:
            report_lines.extend(
                [
                    "",
                    "DETAILED BREAKDOWN",
                    "-" * 40,
                    f"True Positives:  {metrics['true_positives']}",
                    f"True Negatives:  {metrics['true_negatives']}",
                    f"False Positives: {metrics['false_positives']}",
                    f"False Negatives: {metrics['false_negatives']}",
                    f"Specificity:     {metrics['specificity']:.4f}",
                ]
            )

        report_lines.extend(
            [
                "",
                "=" * 60,
                f"Samples Evaluated: {metrics.get('n_samples', 'N/A')}",
                "=" * 60,
            ]
        )

        return "\n".join(report_lines)

    def check_performance_threshold(
        self,
        metrics: Dict[str, Any],
        thresholds: Dict[str, float],
    ) -> Dict[str, bool]:
        """
        Check if metrics meet minimum thresholds.

        Args:
            metrics: Evaluation metrics
            thresholds: Minimum threshold for each metric

        Returns:
            Dict indicating if each threshold is met
        """
        results = {}
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] is not None:
                results[metric] = metrics[metric] >= threshold
            else:
                results[metric] = False
        return results
