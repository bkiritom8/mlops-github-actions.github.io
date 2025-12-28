.PHONY: help venv install setup train evaluate test clean all pipeline clean-venv report

# Python interpreter
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Default target
help:
	@echo "Available targets:"
	@echo "  venv          - Create virtual environment"
	@echo "  install       - Install Python dependencies (run after activating venv)"
	@echo "  setup         - Create venv and setup project environment"
	@echo "  train         - Train the ML model"
	@echo "  validate-data - Validate dataset quality"
	@echo "  evaluate      - Evaluate the trained model"
	@echo "  test          - Run all tests"
	@echo "  pipeline      - Run complete ML pipeline"
	@echo "  report        - Generate project summary report"
	@echo "  clean         - Clean generated files"
	@echo "  clean-venv    - Remove virtual environment"
	@echo "  all           - Run complete pipeline with tests"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make venv"
	@echo "  2. source venv/bin/activate  (or 'venv\\Scripts\\activate' on Windows)"
	@echo "  3. make install"
	@echo "  4. make pipeline"
	@echo "  5. make report        (view project summary)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created!"
	@echo ""
	@echo "To activate, run:"
	@echo "  source $(VENV_BIN)/activate   (macOS/Linux)"
	@echo "  $(VENV)\\Scripts\\activate     (Windows)"

# Install dependencies (assumes venv is activated)
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Setup environment with venv
setup: venv
	@echo "Setting up project..."
	@echo "Activating venv and installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Setting up project directories..."
	mkdir -p data models reports tests
	@echo ""
	@echo "Setup complete!"
	@echo "To activate virtual environment, run:"
	@echo "  source $(VENV_BIN)/activate"

# Train model
train:
	@echo "Training model..."
	python src/train.py

# Validate data
validate-data:
	@echo "Validating data..."
	python src/validate_data.py

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python src/evaluate.py

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

# Complete ML pipeline
pipeline: train validate-data evaluate
	@echo "ML Pipeline completed successfully!"

# Run everything including tests
all: setup pipeline test
	@echo "All tasks completed successfully!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf models/*.joblib
	rm -rf reports/*.png reports/*.json reports/*.txt
	rm -rf data/wine_data.csv
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	@echo "Clean complete!"

# Remove virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Virtual environment removed!"

# Lint code (optional)
lint:
	@echo "Linting code..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

# Format code (optional)
format:
	@echo "Formatting code..."
	black src/ tests/

# Generate project report
report:
	@echo "Generating project report..."
	@python generate_report.py
