"""
ML Model Training Script
Trains a classification model on wine quality dataset
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


def load_params(params_path='params.yaml'):
    """Load training parameters from YAML file"""
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        # Default parameters
        params = {
            'train': {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2
            }
        }
    return params


def prepare_data(test_size=0.2, random_state=42):
    """Load and prepare wine quality dataset"""
    print("Loading wine quality dataset...")

    # Load wine dataset from sklearn
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='quality')

    # Save raw data for reproducibility
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    df = X.copy()
    df['quality'] = y
    df.to_csv(os.path.join(data_dir, 'wine_data.csv'), index=False)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100, max_depth=10,
                min_samples_split=2, random_state=42):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Model training completed!")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    return metrics


def save_artifacts(model, scaler, metrics):
    """Save model, scaler, and metrics"""
    print("\nSaving artifacts...")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Save model and scaler
    model_path = 'models/model.joblib'
    scaler_path = 'models/scaler.joblib'

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # Save metrics
    metrics_path = 'reports/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("Starting ML Training Pipeline")
    print("=" * 50)

    # Load parameters
    params = load_params()
    train_params = params.get('train', {})

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        test_size=train_params.get('test_size', 0.2),
        random_state=train_params.get('random_state', 42)
    )

    # Train model
    model = train_model(
        X_train, y_train,
        n_estimators=train_params.get('n_estimators', 100),
        max_depth=train_params.get('max_depth', 10),
        min_samples_split=train_params.get('min_samples_split', 2),
        random_state=train_params.get('random_state', 42)
    )

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Save artifacts
    save_artifacts(model, scaler, metrics)

    print("\n" + "=" * 50)
    print("Training Pipeline Completed Successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
