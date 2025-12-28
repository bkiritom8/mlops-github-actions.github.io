"""
Generate Project Summary Report
Creates a comprehensive report of the MLOps project
"""
import os
import json
from datetime import datetime
import subprocess


def get_git_info():
    """Get git repository information"""
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return branch, commit
    except:
        return "unknown", "unknown"


def count_files():
    """Count files by type"""
    counts = {
        'Python files': 0,
        'Test files': 0,
        'Workflow files': 0,
        'Total files': 0
    }

    for root, dirs, files in os.walk('.'):
        # Skip virtual env and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']

        for file in files:
            if file.endswith('.py'):
                counts['Python files'] += 1
                if 'test' in file:
                    counts['Test files'] += 1
            elif file.endswith('.yml') or file.endswith('.yaml'):
                counts['Workflow files'] += 1
            counts['Total files'] += 1

    return counts


def get_model_metrics():
    """Get latest model metrics if available"""
    metrics_path = 'reports/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def check_artifacts():
    """Check for generated artifacts"""
    artifacts = {
        'Model file': os.path.exists('models/model.joblib'),
        'Scaler file': os.path.exists('models/scaler.joblib'),
        'Metrics report': os.path.exists('reports/metrics.json'),
        'Confusion matrix': os.path.exists('reports/confusion_matrix.png'),
        'Feature importance': os.path.exists('reports/feature_importance.png'),
        'Classification report': os.path.exists('reports/classification_report.txt')
    }
    return artifacts


def generate_report():
    """Generate comprehensive project report"""

    print("=" * 70)
    print("MLOps CI/CD Pipeline - Project Report".center(70))
    print("=" * 70)
    print()

    # Date and time
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Git information
    branch, commit = get_git_info()
    print("Repository Information")
    print("-" * 70)
    print(f"Branch:        {branch}")
    print(f"Commit:        {commit}")
    print()

    # Project structure
    file_counts = count_files()
    print("Project Structure")
    print("-" * 70)
    for key, value in file_counts.items():
        print(f"{key:20s}: {value}")
    print()

    # Artifacts status
    artifacts = check_artifacts()
    print("Generated Artifacts")
    print("-" * 70)
    for artifact, exists in artifacts.items():
        status = "✓ Found" if exists else "✗ Missing"
        print(f"{artifact:30s}: {status}")
    print()

    # Model metrics
    metrics = get_model_metrics()
    if metrics:
        print("Model Performance Metrics")
        print("-" * 70)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric.capitalize():20s}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{metric.capitalize():20s}: {value}")
        print()
    else:
        print("Model Performance Metrics")
        print("-" * 70)
        print("No metrics found. Run 'make train' to generate metrics.")
        print()

    # Key features
    print("Key Features")
    print("-" * 70)
    features = [
        "✓ Automated CI/CD pipeline with GitHub Actions",
        "✓ Continuous Machine Learning (CML) integration",
        "✓ Comprehensive data validation and quality checks",
        "✓ Automated model training and evaluation",
        "✓ Unit and integration testing with pytest",
        "✓ Model performance visualization",
        "✓ Makefile automation for common tasks",
        "✓ Virtual environment support",
        "✓ Deployment simulation workflow"
    ]
    for feature in features:
        print(f"  {feature}")
    print()

    # Technologies
    print("Technology Stack")
    print("-" * 70)
    technologies = {
        "Languages": "Python 3.10+",
        "ML Framework": "scikit-learn, pandas, numpy",
        "Testing": "pytest, pytest-cov",
        "CI/CD": "GitHub Actions, CML",
        "Visualization": "matplotlib, seaborn",
        "Automation": "Make, Bash",
        "Version Control": "Git, GitHub"
    }
    for tech, tools in technologies.items():
        print(f"{tech:20s}: {tools}")
    print()

    # Quick start
    print("Quick Start Commands")
    print("-" * 70)
    commands = [
        ("Setup project", "make setup"),
        ("Activate venv", "source venv/bin/activate"),
        ("Run pipeline", "make pipeline"),
        ("Run tests", "make test"),
        ("Clean files", "make clean"),
        ("View help", "make help")
    ]
    for desc, cmd in commands:
        print(f"{desc:20s}: {cmd}")
    print()

    # Project status
    print("Project Status")
    print("-" * 70)

    # Check if trained
    if artifacts['Model file']:
        print("Status: ✓ Model trained and ready")
        if metrics and metrics.get('accuracy', 0) >= 0.80:
            print(f"Quality: ✓ Meets performance threshold (≥80%)")
        else:
            print(f"Quality: ⚠ Below performance threshold (<80%)")
    else:
        print("Status: ○ Model not trained yet")
        print("Action: Run 'make train' to train the model")
    print()

    # Next steps
    print("Next Steps")
    print("-" * 70)
    if not artifacts['Model file']:
        print("1. Run 'make pipeline' to train and evaluate the model")
        print("2. Run 'make test' to verify everything works")
        print("3. Create a PR to see automated CML reports")
    else:
        print("1. Create a feature branch: git checkout -b feature/improve-model")
        print("2. Modify params.yaml to experiment with hyperparameters")
        print("3. Create a PR to see automated comparison reports")
        print("4. Check GitHub Actions for CI/CD pipeline status")
    print()

    print("=" * 70)
    print("Report complete!".center(70))
    print("=" * 70)
    print()
    print("For more information:")
    print("  - View README.md for full documentation")
    print("  - Check SHOWCASE.md for presentation tips")
    print("  - See ARCHITECTURE.md for system design")
    print("  - Run demo.ipynb for interactive exploration")
    print()


if __name__ == '__main__':
    generate_report()
