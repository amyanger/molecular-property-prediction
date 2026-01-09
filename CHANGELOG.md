# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024

### Added
- **Models**
  - MLP model with Morgan fingerprints
  - GNN model with graph convolutional layers
  - AttentiveFP model with attention mechanism
  - Ensemble model combining all three approaches

- **Utilities**
  - SMILES validation and standardization (`src/utils/smiles.py`)
  - Molecular visualization utilities (`src/utils/visualization.py`)
  - Model evaluation metrics (`src/utils/metrics.py`)
  - Cross-validation utilities (`src/utils/cross_validation.py`)
  - Model interpretability tools (`src/utils/interpretability.py`)
  - Data augmentation utilities (`src/utils/augmentation.py`)
  - Model checkpointing (`src/utils/checkpointing.py`)
  - Configuration management (`src/utils/config.py`)
  - Logging utilities (`src/utils/logger.py`)

- **Scripts**
  - Training scripts for all models
  - Batch prediction script
  - Command-line interface (CLI)
  - FastAPI REST endpoint

- **Infrastructure**
  - Streamlit dashboard for interactive predictions
  - Docker support with multi-stage build
  - GitHub Actions CI/CD pipeline
  - Pre-commit hooks configuration
  - Comprehensive test suite

- **Documentation**
  - README with project overview
  - Training guide
  - Improvement guide
  - Contributing guidelines

### Performance
- Achieved 0.867 AUC-ROC on Tox21 test set
- Ensemble outperforms individual models
- Supports GPU acceleration with CUDA

## [Unreleased]

### Planned
- ONNX model export
- Additional model architectures
- Hyperparameter optimization
- Extended documentation
