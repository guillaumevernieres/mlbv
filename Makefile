.PHONY: help install dev test clean lint format train
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install for development
	pip install -e .[dev]

test: ## Run tests
	python -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage
	python -m pytest tests/ -v --cov=icenet --cov-report=html --cov-report=term

test-hpc: ## Test HPC setup
	python tests/test_hpc_setup.py --gpus 0

lint: ## Run linting
	flake8 icenet/ scripts/ tests/

format: ## Format code with black
	black icenet/ scripts/ tests/

train: ## Run basic training with sample data
	python scripts/train.py --create-data --config configs/config.yaml

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
