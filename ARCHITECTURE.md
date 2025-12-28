# MLOps Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Developer Workflow                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ├─── Code Changes
                                 ├─── Parameter Updates
                                 └─── New Features
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Git Repository                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Source     │  │   Tests      │  │  Workflows   │          │
│  │   Code       │  │              │  │   (YAML)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                      Push / Pull Request
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GitHub Actions CI/CD                       │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 1: Environment Setup                                │  │
│  │  • Checkout code                                          │  │
│  │  • Setup Python 3.10+                                     │  │
│  │  • Install dependencies                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 2: Data Validation                                  │  │
│  │  • Check data quality                                     │  │
│  │  • Validate schema                                        │  │
│  │  • Detect missing values                                  │  │
│  │  • Check data ranges                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 3: Model Training                                   │  │
│  │  • Load dataset                                           │  │
│  │  • Preprocess features                                    │  │
│  │  • Train Random Forest                                    │  │
│  │  • Save model artifacts                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 4: Model Evaluation                                 │  │
│  │  • Calculate metrics                                      │  │
│  │  • Generate confusion matrix                              │  │
│  │  • Create visualizations                                  │  │
│  │  • Check performance threshold                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 5: Automated Testing                                │  │
│  │  • Run unit tests                                         │  │
│  │  • Integration tests                                      │  │
│  │  • Quality checks                                         │  │
│  │  • Coverage report                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 6: CML Report (if PR)                               │  │
│  │  • Generate markdown report                               │  │
│  │  • Upload visualizations                                  │  │
│  │  • Post comment to PR                                     │  │
│  │  • Show metrics comparison                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 7: Artifact Storage                                 │  │
│  │  • Upload model files                                     │  │
│  │  • Save reports                                           │  │
│  │  • Store metrics                                          │  │
│  │  • Archive logs                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                  ┌───────────────┴────────────────┐
                  │                                │
                  ▼                                ▼
    ┌─────────────────────────┐    ┌─────────────────────────┐
    │   Deployment (main)     │    │   PR Comment Report     │
    │  • Create release tag   │    │  • Metrics dashboard    │
    │  • Upload to registry   │    │  • Visualizations       │
    │  • Simulate production  │    │  • Performance summary  │
    └─────────────────────────┘    └─────────────────────────┘
```

## Component Details

### 1. Source Code (`src/`)
- **train.py**: Main training script
  - Loads wine quality dataset
  - Trains Random Forest classifier
  - Saves model and scaler
  - Logs metrics

- **evaluate.py**: Model evaluation
  - Loads trained model
  - Generates predictions
  - Creates visualizations
  - Validates performance

- **validate_data.py**: Data quality checks
  - Schema validation
  - Missing value detection
  - Data type verification
  - Outlier detection

- **predict.py**: Inference script
  - Loads model artifacts
  - Makes predictions
  - Returns class probabilities

### 2. Tests (`tests/`)
- **test_model.py**: Model quality tests
  - Training pipeline validation
  - Prediction functionality
  - Performance thresholds
  - Artifact verification

- **test_data.py**: Data quality tests
  - Schema validation
  - Distribution checks
  - Feature variance
  - Sample size verification

### 3. GitHub Actions Workflows

#### ML Pipeline (`ml-pipeline.yml`)
```yaml
Triggers: Push to main/claude/**, PRs to main
Steps:
  1. Setup environment
  2. Validate data
  3. Train model
  4. Evaluate model
  5. Run tests
  6. Generate CML report (PR only)
  7. Upload artifacts
  8. Check thresholds
```

#### Model Deployment (`model-deployment.yml`)
```yaml
Triggers: Successful ML pipeline on main
Steps:
  1. Download artifacts
  2. Simulate deployment
  3. Create release tag
```

### 4. Configuration Files

- **params.yaml**: Training parameters
  - Model hyperparameters
  - Data split ratios
  - Random seeds
  - Performance thresholds

- **requirements.txt**: Python dependencies
  - ML libraries (scikit-learn, pandas, numpy)
  - Visualization (matplotlib, seaborn)
  - Testing (pytest, pytest-cov)
  - Utilities (pyyaml, joblib)

- **Makefile**: Task automation
  - Virtual environment setup
  - Dependency installation
  - Pipeline execution
  - Testing and cleaning

## Data Flow

```
┌──────────────┐
│  Raw Data    │
│  (sklearn)   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Data Loading    │
│  & Preprocessing │
└──────┬───────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌─────────────┐
│  Training   │      │    Test     │
│    Set      │      │     Set     │
└──────┬──────┘      └──────┬──────┘
       │                    │
       │                    │
       ▼                    │
┌─────────────┐             │
│   Scaler    │             │
│  (fit)      │             │
└──────┬──────┘             │
       │                    │
       ▼                    ▼
┌─────────────────────────────┐
│   Scaled Features           │
│   (Train & Test)            │
└──────┬──────────────────────┘
       │
       ├────────────┬─────────────┐
       │            │             │
       ▼            ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│  Model  │  │  Metrics │  │  Plots   │
│ (.pkl)  │  │ (.json)  │  │  (.png)  │
└─────────┘  └──────────┘  └──────────┘
```

## Technology Stack

### Core ML
- **scikit-learn**: Model training and evaluation
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualizations

### Testing
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting

### DevOps
- **GitHub Actions**: CI/CD automation
- **CML**: ML reporting
- **Make**: Task automation
- **Git**: Version control

### Utilities
- **joblib**: Model serialization
- **pyyaml**: Configuration management
- **python-dotenv**: Environment variables

## Performance Metrics

The pipeline tracks:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision/recall

Minimum threshold: **80% accuracy**

## Deployment Simulation

Current: GitHub Actions artifact upload
Future extensions:
- Docker containerization
- Cloud deployment (AWS/Azure/GCP)
- API endpoint (FastAPI/Flask)
- Model registry (MLflow)
- Monitoring (Prometheus/Grafana)

## Security Considerations

- No hardcoded credentials
- Environment variables for secrets
- Minimal dependencies
- Regular security updates
- Test coverage for edge cases

## Scalability

The architecture supports:
- Multiple models
- Different datasets
- Parallel experiments
- A/B testing
- Model versioning
- Distributed training (future)

## Monitoring & Logging

- GitHub Actions logs
- Test coverage reports
- Performance metrics tracking
- CML visualization reports
- Artifact versioning
