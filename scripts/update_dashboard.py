#!/usr/bin/env python3
"""Script to update the GitHub Pages dashboard data."""

import argparse
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.metrics_reporter import MetricsReporter


def get_git_info() -> dict:
    """Get current git information."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        return {
            "commit": commit,
            "branch": branch,
            "short_commit": commit[:7],
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="Update dashboard data")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="models/artifacts",
        help="Directory containing pipeline artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/assets",
        help="Output directory for dashboard data",
    )

    args = parser.parse_args()

    reporter = MetricsReporter(output_dir=args.output_dir)

    # Collect metrics
    print("Collecting pipeline metrics...")
    metrics = reporter.collect_pipeline_metrics(args.artifacts_dir)

    # Get git info
    git_info = get_git_info()

    # Generate dashboard data
    print("Generating dashboard data...")
    dashboard_data = reporter.generate_dashboard_data(metrics, git_info)

    # Save dashboard data
    output_path = reporter.save_dashboard_data(dashboard_data)
    print(f"Dashboard data saved to: {output_path}")

    # Print summary
    if metrics.get("latest_run"):
        print(f"\nLatest Run Summary:")
        print(f"  Run ID: {metrics['latest_run'].get('run_id')}")
        print(f"  Status: {metrics['latest_run'].get('status')}")
        print(f"  Accuracy: {metrics['latest_run'].get('accuracy', 0):.4f}")
        print(f"  F1 Score: {metrics['latest_run'].get('f1_score', 0):.4f}")
    else:
        print("\nNo pipeline runs found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
