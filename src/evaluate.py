"""
Model Evaluation Script
Evaluates trained model and generates reports
"""
import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import joblib
import yaml


def load_params(params_path='params.yaml'):
    """Load parameters from YAML file"""
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = {'data': {'min_accuracy_threshold': 0.85}}
    return params


def load_artifacts():
    """Load model, scaler, and test data"""
    print("Loading model artifacts...")

    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')

    print("✓ Model and scaler loaded successfully")
    return model, scaler


def load_test_data():
    """Load and prepare test data"""
    print("Loading test data...")

    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    # Load wine dataset
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='quality')

    # Use same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Loaded {len(X_test)} test samples")
    return X_test, y_test


def generate_confusion_matrix_plot(y_true, y_pred, classes):
    """Generate and save confusion matrix plot"""
    print("Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plot_path = 'reports/confusion_matrix.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to {plot_path}")


def generate_feature_importance_plot(model, feature_names):
    """Generate and save feature importance plot"""
    print("Generating feature importance plot...")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features

        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)),
                   [feature_names[i] for i in indices],
                   rotation=45, ha='right')
        plt.tight_layout()

        plot_path = 'reports/feature_importance.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✓ Feature importance plot saved to {plot_path}")


def check_model_performance(metrics, threshold=0.85):
    """Check if model meets minimum performance threshold"""
    print(f"\nChecking model performance against threshold ({threshold})...")

    accuracy = metrics['accuracy']

    if accuracy >= threshold:
        print(f"✓ Model passed: Accuracy {accuracy:.4f} >= {threshold}")
        return True
    else:
        print(f"✗ Model failed: Accuracy {accuracy:.4f} < {threshold}")
        return False


def evaluate():
    """Main evaluation pipeline"""
    print("=" * 50)
    print("Starting Model Evaluation")
    print("=" * 50)

    # Load parameters
    params = load_params()
    threshold = params.get('data', {}).get('min_accuracy_threshold', 0.85)

    # Load artifacts
    model, scaler = load_artifacts()
    X_test, y_test = load_test_data()

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)

    # Load existing metrics
    metrics_path = 'reports/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics['test_accuracy'] = float(accuracy)

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate visualizations
    print("\nGenerating evaluation reports...")
    os.makedirs('reports', exist_ok=True)

    from sklearn.datasets import load_wine
    wine = load_wine()

    generate_confusion_matrix_plot(y_test, y_pred, wine.target_names)
    generate_feature_importance_plot(model, wine.feature_names)

    # Generate classification report
    report = classification_report(y_test, y_pred,
                                    target_names=wine.target_names)
    print("\nClassification Report:")
    print(report)

    # Save classification report
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)

    # Check performance threshold
    passed = check_model_performance(metrics, threshold)

    print("\n" + "=" * 50)
    print("Model Evaluation Completed!")
    print("=" * 50)

    return passed


if __name__ == '__main__':
    passed = evaluate()
    sys.exit(0 if passed else 1)
