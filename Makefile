# Molecular Property Prediction - Makefile
# Common commands for development and deployment

.PHONY: help install install-dev test lint format clean train train-all app docker docker-run download

# Default target
help:
	@echo "Molecular Property Prediction - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make download       Download Tox21 dataset"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run unit tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code with ruff"
	@echo "  make clean          Remove build artifacts and cache"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train MLP model"
	@echo "  make train-gnn      Train GNN model"
	@echo "  make train-afp      Train AttentiveFP model"
	@echo "  make train-all      Train all models sequentially"
	@echo "  make ensemble       Evaluate ensemble model"
	@echo ""
	@echo "Application:"
	@echo "  make app            Run Streamlit dashboard"
	@echo "  make api            Run FastAPI prediction server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker         Build Docker image"
	@echo "  make docker-run     Run Docker container"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff mypy

download:
	python scripts/download_data.py

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .ruff_cache
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Training
# =============================================================================

train:
	python scripts/train.py --epochs 50 --batch_size 64

train-gnn:
	python scripts/train_gnn.py --epochs 50 --batch_size 64

train-afp:
	python scripts/train_attentivefp.py --epochs 100 --batch_size 64

train-all:
	@echo "Training MLP..."
	python scripts/train.py --epochs 50
	@echo ""
	@echo "Training GNN..."
	python scripts/train_gnn.py --epochs 50
	@echo ""
	@echo "Training AttentiveFP..."
	python scripts/train_attentivefp.py --epochs 100
	@echo ""
	@echo "All models trained successfully!"

ensemble:
	python scripts/ensemble_all.py

# =============================================================================
# Application
# =============================================================================

app:
	streamlit run app.py

api:
	uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Docker
# =============================================================================

docker:
	docker build -t molprop:latest .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/models:/app/models molprop:latest

docker-train:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models \
		--entrypoint python molprop:latest scripts/train.py

# =============================================================================
# Utilities
# =============================================================================

predict:
	@read -p "Enter SMILES string: " smiles; \
	python scripts/predict.py --smiles "$$smiles"

compare:
	python scripts/compare_models.py

visualize:
	python scripts/visualize_results.py
