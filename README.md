# MLOps CI/CD Pipeline with GitHub Actions

A complete end-to-end MLOps project demonstrating automated machine learning workflows using GitHub Actions, CML (Continuous Machine Learning), and Makefile automation.

## Project Overview

This project automates the entire ML workflow from data validation to model deployment. Every code change automatically triggers:
- Data quality validation
- Model training
- Performance evaluation
- Automated testing
- Model deployment (simulation)

## What You'll Learn

- **CI/CD for ML**: Automate ML pipelines using GitHub Actions
- **CML Integration**: Generate model reports and visualizations in pull requests
- **Workflow Automation**: Use Makefiles to simplify complex ML tasks
- **MLOps Best Practices**: Testing, validation, and reproducible ML pipelines
- **Real-time Automation**: Automatic retraining and deployment on code changes

## Project Structure

```
mlops-github-actions/
├── .github/
│   └── workflows/
│       ├── ml-pipeline.yml          # Main CI/CD workflow
│       └── model-deployment.yml     # Deployment workflow
├── data/                            # Dataset storage
├── models/                          # Trained model artifacts
├── reports/                         # Evaluation reports and metrics
├── src/
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Model evaluation script
│   └── validate_data.py            # Data validation script
├── tests/
│   ├── test_model.py               # Model tests
│   └── test_data.py                # Data quality tests
├── Makefile                         # Automation commands
├── params.yaml                      # Training parameters
└── requirements.txt                 # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Make (optional, but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mlops-github-actions
   ```

2. **Install dependencies**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

## Usage

### Using Makefile (Recommended)

The Makefile provides convenient commands for all common tasks:

```bash
# View all available commands
make help

# Setup project
make setup

# Run complete ML pipeline
make pipeline

# Individual steps
make validate-data    # Validate dataset
make train           # Train model
make evaluate        # Evaluate model
make test            # Run tests

# Clean generated files
make clean

# Run everything (setup + pipeline + tests)
make all
```

### Manual Execution

You can also run scripts individually:

```bash
# Validate data
python src/validate_data.py

# Train model
python src/train.py

# Evaluate model
python src/evaluate.py

# Run tests
pytest tests/ -v
```

## CI/CD Pipeline

### How It Works

The automated pipeline triggers on every push or pull request:

1. **Data Validation**
   - Checks data quality, missing values, and schema
   - Validates data types and ranges
   - Ensures minimum sample requirements

2. **Model Training**
   - Trains Random Forest classifier on wine quality dataset
   - Uses parameters from `params.yaml`
   - Saves model artifacts to `models/`

3. **Model Evaluation**
   - Evaluates model on test set
   - Generates confusion matrix and feature importance plots
   - Creates classification report
   - Saves metrics to `reports/metrics.json`

4. **Automated Testing**
   - Runs unit tests for data and model quality
   - Checks model meets accuracy threshold (80%)
   - Validates predictions and artifacts

5. **CML Reporting** (on Pull Requests)
   - Posts detailed report as PR comment
   - Includes metrics, plots, and classification results
   - Enables easy model comparison

6. **Model Deployment** (on main branch)
   - Uploads model artifacts
   - Creates deployment tags
   - Simulates production deployment

### GitHub Actions Workflows

#### ML Pipeline (`ml-pipeline.yml`)
- **Triggers**: Push to main or claude/** branches, PRs to main
- **Steps**: Data validation → Training → Evaluation → Testing → CML Report
- **Outputs**: Model artifacts, metrics, visualizations

#### Model Deployment (`model-deployment.yml`)
- **Triggers**: After successful ML pipeline on main branch
- **Steps**: Download artifacts → Deploy → Tag release

## Model Details

### Dataset
- **Source**: Scikit-learn wine quality dataset
- **Task**: Multi-class classification
- **Classes**: 3 wine quality classes
- **Features**: 13 chemical properties

### Model
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: Configured in `params.yaml`
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

### Performance Threshold
- Minimum accuracy: 80%
- Pipeline fails if model doesn't meet threshold
- Prevents poor models from being deployed

## Configuration

### Training Parameters (`params.yaml`)

```yaml
train:
  test_size: 0.2              # Train/test split ratio
  random_state: 42            # Reproducibility seed
  n_estimators: 100           # Number of trees
  max_depth: 10               # Max tree depth
  min_samples_split: 2        # Min samples for split

data:
  min_accuracy_threshold: 0.85  # Minimum acceptable accuracy
```

### Modifying Parameters

1. Edit `params.yaml`
2. Commit and push changes
3. Pipeline automatically retrains with new parameters
4. Compare results in PR comment

## Testing

### Test Coverage

The project includes comprehensive tests:

- **Data Tests** (`test_data.py`)
  - Data shape and size validation
  - Missing value checks
  - Feature variance validation
  - Outlier detection

- **Model Tests** (`test_model.py`)
  - Training pipeline validation
  - Model quality checks
  - Prediction functionality
  - Accuracy threshold validation

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Outputs and Artifacts

### Model Artifacts
- `models/model.joblib`: Trained Random Forest model
- `models/scaler.joblib`: Feature scaler

### Reports
- `reports/metrics.json`: Performance metrics
- `reports/confusion_matrix.png`: Confusion matrix visualization
- `reports/feature_importance.png`: Feature importance plot
- `reports/classification_report.txt`: Detailed classification metrics

### Data
- `data/wine_data.csv`: Processed wine dataset

## Real-World Applications

This pipeline demonstrates concepts used in production ML systems:

### Current Implementation
- Automated training and evaluation
- Data quality validation
- Model performance monitoring
- Reproducible experiments

### Production Extensions
- **Cloud Deployment**: Deploy to AWS SageMaker, Azure ML, or GCP Vertex AI
- **Model Registry**: Use MLflow or similar for model versioning
- **Monitoring**: Add model drift detection and performance monitoring
- **A/B Testing**: Compare model versions in production
- **Feature Store**: Centralized feature management
- **Data Versioning**: Use DVC for dataset versioning

## Troubleshooting

### Common Issues

1. **Import errors**
   ```bash
   # Reinstall dependencies
   make install
   ```

2. **Test failures**
   ```bash
   # Check if model is trained
   make train
   # Then run tests
   make test
   ```

3. **GitHub Actions failures**
   - Check workflow logs in Actions tab
   - Verify all required files are committed
   - Ensure tests pass locally first

## Best Practices Demonstrated

- **Version Control**: All code, configs, and workflows in Git
- **Reproducibility**: Fixed random seeds and versioned dependencies
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Clear README and code comments
- **Automation**: One-command pipeline execution
- **Continuous Integration**: Automated testing on every commit
- **Model Validation**: Performance thresholds prevent bad deployments

## Learning Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CML Documentation](https://cml.dev/)
- [MLOps Principles](https://ml-ops.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## Next Steps

To extend this project:

1. **Add More Models**: Try different algorithms (XGBoost, Neural Networks)
2. **Hyperparameter Tuning**: Implement grid search or Bayesian optimization
3. **Feature Engineering**: Create new features and test impact
4. **Real Deployment**: Deploy to cloud platform (AWS, Azure, GCP)
5. **Monitoring**: Add model monitoring and retraining triggers
6. **Data Versioning**: Integrate DVC for dataset tracking
7. **Experiment Tracking**: Add MLflow for experiment management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request
6. Review automated reports in PR comments

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Scikit-learn wine quality dataset
- Tools: GitHub Actions, CML, Scikit-learn, Pytest
- Inspiration: MLOps community best practices

---

**Built with ❤️ for learning MLOps**
