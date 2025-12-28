.PHONY: help install setup train evaluate test clean all pipeline

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Setup project environment"
	@echo "  train         - Train the ML model"
	@echo "  validate-data - Validate dataset quality"
	@echo "  evaluate      - Evaluate the trained model"
	@echo "  test          - Run all tests"
	@echo "  pipeline      - Run complete ML pipeline"
	@echo "  clean         - Clean generated files"
	@echo "  all           - Run complete pipeline with tests"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Setup environment
setup: install
	@echo "Setting up project directories..."
	mkdir -p data models reports tests
	@echo "Setup complete!"

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
pipeline: validate-data train evaluate
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

# Lint code (optional)
lint:
	@echo "Linting code..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

# Format code (optional)
format:
	@echo "Formatting code..."
	black src/ tests/
