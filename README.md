# Molecular Property Prediction

Deep learning models for predicting molecular toxicity across 12 biological endpoints. Trained on the Tox21 benchmark dataset used in pharmaceutical drug discovery.

## What This Does

Given a molecule's structure (SMILES notation), the model predicts:
- **Toxicity risk** across 12 biological targets
- **Hormone disruption** (Androgen/Estrogen receptors)
- **Cancer risk** (p53 tumor suppressor pathway)
- **Cellular stress** (Mitochondrial toxicity, heat shock response)

### Example Predictions

| Molecule | Toxicity Score | Risk Level |
|----------|---------------|------------|
| Aspirin | 1.6% | LOW |
| Caffeine | 0.9% | LOW |
| DDT (pesticide) | 80.7% | HIGH |

## Results

### Model Performance (Tox21 Benchmark)

| Model | Test AUC-ROC | Notes |
|-------|-------------|-------|
| **MLP** | **0.810** | Best for fingerprint inputs |
| Transformer | 0.724 | Better suited for sequence inputs |
| Random Forest | ~0.750 | Traditional ML baseline |
| State-of-the-Art | ~0.850 | Graph Neural Networks |

### Per-Task Performance (MLP)

| Endpoint | AUC-ROC | Description |
|----------|---------|-------------|
| NR-AhR | 0.873 | Dioxin-like toxicity |
| SR-MMP | 0.849 | Mitochondrial toxicity |
| NR-AR-LBD | 0.842 | Androgen receptor binding |
| SR-p53 | 0.839 | Cancer risk (tumor suppressor) |
| SR-ATAD5 | 0.838 | DNA damage response |
| NR-Aromatase | 0.827 | Estrogen synthesis disruption |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/amyanger/molecular-property-prediction.git
cd molecular-property-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install rdkit pandas scikit-learn matplotlib tqdm

# Download datasets
python scripts/download_data.py

# Train model
python scripts/train.py --model mlp --epochs 30

# Run predictions
python scripts/predict.py
```

## Project Structure

```
molecular-property-prediction/
├── scripts/
│   ├── train.py              # Model training
│   ├── predict.py            # Inference on new molecules
│   ├── download_data.py      # Dataset downloader
│   └── visualize_results.py  # Performance visualization
├── models/
│   └── results.json          # Training metrics
├── data/                     # Datasets (downloaded separately)
├── configs/                  # Hyperparameters
└── notebooks/                # Exploration notebooks
```

## Technical Details

### Model Architecture

**MLP Model:**
- Input: 2048-bit Morgan fingerprints (ECFP4)
- Hidden layers: 1024 → 512 → 256
- Multi-task output: 12 toxicity endpoints
- Dropout: 0.3, BatchNorm, AdamW optimizer

**Transformer Model:**
- Fingerprint projection to 256-dim embeddings
- 4-layer transformer encoder
- 8 attention heads
- GELU activation

### Dataset

**Tox21** - NIH toxicity screening data
- 7,831 compounds
- 12 biological assay endpoints
- Multi-label classification

## Hardware

Trained on NVIDIA RTX 5090 (32GB VRAM) with PyTorch 2.11 nightly + CUDA 12.8

## References

- [Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/)
- [MoleculeNet Benchmark](https://moleculenet.org/)
- [DeepChem](https://deepchem.io/)

## License

MIT License
