"""
Model Prediction Script
Make predictions using trained model
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_wine


def load_model_artifacts():
    """Load trained model and scaler"""
    model_path = 'models/model.joblib'
    scaler_path = 'models/scaler.joblib'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first using 'make train'"
        )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Please train the model first using 'make train'"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def predict_sample(model, scaler, features):
    """Make prediction for a single sample"""
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return prediction, probability


def main():
    """Main prediction pipeline"""
    print("=" * 50)
    print("Wine Quality Prediction")
    print("=" * 50)

    # Load model and scaler
    print("\nLoading model artifacts...")
    model, scaler = load_model_artifacts()
    print("✓ Model loaded successfully")

    # Load wine dataset for class names
    wine = load_wine()
    class_names = wine.target_names
    feature_names = wine.feature_names

    # Example: predict on a few samples
    print("\nMaking sample predictions...")
    print("-" * 50)

    # Use some samples from the dataset
    X = wine.data[:5]

    for i, features in enumerate(X):
        prediction, probabilities = predict_sample(model, scaler, features)

        print(f"\nSample {i + 1}:")
        print(f"  Predicted class: {class_names[prediction]}")
        print(f"  Confidence: {probabilities[prediction]:.2%}")
        print(f"  Probabilities:")
        for class_idx, prob in enumerate(probabilities):
            print(f"    - {class_names[class_idx]}: {prob:.2%}")

    print("\n" + "=" * 50)
    print("Predictions completed!")
    print("=" * 50)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
