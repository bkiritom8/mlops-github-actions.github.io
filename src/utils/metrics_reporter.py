"""Metrics reporting utilities for dashboard and monitoring."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


class MetricsReporter:
    """Generate metrics reports for GitHub Pages dashboard."""

    def __init__(self, output_dir: str = "docs/assets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_pipeline_metrics(self, artifacts_dir: str = "models/artifacts") -> Dict[str, Any]:
        """
        Collect all pipeline metrics from artifacts directory.

        Args:
            artifacts_dir: Directory containing pipeline artifacts

        Returns:
            Aggregated metrics dictionary
        """
        artifacts_path = Path(artifacts_dir)
        metrics = {
            "last_updated": datetime.now().isoformat(),
            "runs": [],
            "latest_run": None,
        }

        # Find all pipeline results
        result_files = sorted(
            artifacts_path.glob("pipeline_results_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for result_file in result_files:
            try:
                with open(result_file, "r") as f:
                    run_data = json.load(f)
                    metrics["runs"].append(
                        {
                            "run_id": run_data.get("run_id"),
                            "status": run_data.get("status"),
                            "timestamp": run_data.get("timestamp"),
                            "accuracy": run_data.get("evaluation_metrics", {}).get("accuracy"),
                            "f1_score": run_data.get("evaluation_metrics", {}).get("f1_score"),
                            "roc_auc": run_data.get("evaluation_metrics", {}).get("roc_auc"),
                        }
                    )
            except (json.JSONDecodeError, IOError):
                continue

        if metrics["runs"]:
            metrics["latest_run"] = metrics["runs"][0]
            metrics["total_runs"] = len(metrics["runs"])

        return metrics

    def generate_dashboard_data(
        self,
        pipeline_metrics: Optional[Dict[str, Any]] = None,
        git_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate data for the GitHub Pages dashboard.

        Args:
            pipeline_metrics: Optional pre-collected metrics
            git_info: Optional git information

        Returns:
            Dashboard data dictionary
        """
        if pipeline_metrics is None:
            pipeline_metrics = self.collect_pipeline_metrics()

        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "project": {
                "name": "MLOps GitHub Actions",
                "version": "1.0.0",
                "description": "End-to-end MLOps demonstration with GitHub Actions",
            },
            "pipeline": pipeline_metrics,
            "features": {
                "data_validation": True,
                "automated_training": True,
                "model_evaluation": True,
                "experiment_tracking": True,
                "ci_cd": True,
                "model_serving": True,
                "monitoring": True,
            },
        }

        if git_info:
            dashboard_data["git"] = git_info

        # Get environment info
        dashboard_data["environment"] = {
            "python_version": os.environ.get("PYTHON_VERSION", "3.9+"),
            "runner": os.environ.get("GITHUB_RUNNER_OS", "ubuntu-latest"),
        }

        return dashboard_data

    def save_dashboard_data(self, data: Dict[str, Any]) -> Path:
        """Save dashboard data to JSON file."""
        output_path = self.output_dir / "dashboard_data.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path

    def generate_metrics_history(self, max_runs: int = 50) -> List[Dict[str, Any]]:
        """Generate metrics history for trend charts."""
        metrics = self.collect_pipeline_metrics()
        return metrics.get("runs", [])[:max_runs]

    def generate_badge_data(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Generate data for status badges.

        Args:
            metrics: Pipeline metrics

        Returns:
            Badge data dictionary
        """
        latest = metrics.get("latest_run", {})

        badges = {
            "pipeline_status": {
                "label": "Pipeline",
                "message": latest.get("status", "unknown"),
                "color": "green" if latest.get("status") == "completed" else "red",
            },
            "accuracy": {
                "label": "Accuracy",
                "message": f"{latest.get('accuracy', 0) * 100:.1f}%",
                "color": self._get_metric_color(latest.get("accuracy", 0)),
            },
            "f1_score": {
                "label": "F1 Score",
                "message": f"{latest.get('f1_score', 0) * 100:.1f}%",
                "color": self._get_metric_color(latest.get("f1_score", 0)),
            },
        }

        return badges

    def _get_metric_color(self, value: float) -> str:
        """Get badge color based on metric value."""
        if value >= 0.9:
            return "brightgreen"
        elif value >= 0.8:
            return "green"
        elif value >= 0.7:
            return "yellow"
        elif value >= 0.6:
            return "orange"
        else:
            return "red"
