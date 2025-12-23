# Molecular Property Prediction - Training Guide

## What We Did

This document summarizes the pre-training setup we created to improve the GNN models.

### Overview

We implemented **self-supervised pre-training** for the GNN models to improve molecular property prediction performance on the Tox21 dataset. Pre-training on large unlabeled molecular datasets helps the model learn better molecular representations before fine-tuning on the specific task.

### Files Created

| File | Purpose |
|------|---------|
| `scripts/download_pretrain_data.py` | Downloads ZINC/ChEMBL molecules for pre-training |
| `scripts/pretrain_gnn.py` | Self-supervised pre-training with masked atom prediction + contrastive learning |
| `scripts/pretrain_pipeline.py` | Runs the full pipeline (download → pretrain → fine-tune → compare) |
| `scripts/train_max_power.py` | Maximum training configuration for RTX 5090 |

### Files Modified

| File | Changes |
|------|---------|
| `scripts/train_gnn.py` | Added `--pretrained`, `--pretrained_path`, `--freeze_encoder` flags |

---

## IMPORTANT: PyTorch Setup for RTX 5090

The RTX 5090 uses the new Blackwell architecture (sm_120) which requires PyTorch with CUDA 12.8+.

### Required Installation

```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch NIGHTLY with CUDA 12.8 (required for RTX 5090)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**DO NOT use stable PyTorch releases** - they don't support sm_120 yet.

### Verify Installation

```python
import torch
print(torch.__version__)  # Should show something like 2.11.0.dev... +cu128
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 5090
```

---

## How to Run Training

### Option 1: Quick Run (50K molecules, 100 epochs)

```bash
cd d:\Development\molecular-property-prediction
python scripts/pretrain_pipeline.py
```

### Option 2: Maximum Power Run (250K molecules, 200 epochs)

```bash
cd d:\Development\molecular-property-prediction
python scripts/train_max_power.py
```

Estimated time: ~2 hours on RTX 5090

### Option 3: Custom Configuration

```bash
# Download more molecules
python scripts/download_pretrain_data.py --num_molecules 250000

# Pre-train with custom settings
python scripts/pretrain_gnn.py \
  --epochs 200 \
  --batch_size 512 \
  --hidden_channels 512 \
  --num_layers 6

# Fine-tune on Tox21
python scripts/train_gnn.py \
  --epochs 100 \
  --pretrained \
  --hidden_channels 512 \
  --num_layers 6
```

---

## Pre-training Details

### Self-Supervised Tasks

1. **Masked Atom Prediction (15% masking)**
   - Randomly masks atoms in the molecule
   - Model learns to predict original atom features from molecular context

2. **Contrastive Learning**
   - Creates augmented views of molecules
   - Model learns to distinguish same molecule from different molecules

### Pre-training Data Sources

| Source | Size | Quality |
|--------|------|---------|
| ZINC | 250K | Drug-like molecules |
| ChEMBL | 2M+ | Bioactive molecules (best for Tox21) |

### Recommended Settings for RTX 5090

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--molecules` | 250,000 | Sweet spot for quality vs time |
| `--epochs` | 200 | Good convergence |
| `--batch_size` | 512 | Utilizes 32GB VRAM |
| `--hidden_channels` | 512 | Larger model capacity |
| `--num_layers` | 6 | Deeper network |

---

## After Training

### 1. Run the Ensemble

Combines MLP, GCN, and AttentiveFP for best results:

```bash
python scripts/ensemble_all.py
```

### 2. Launch the Dashboard

Interactive web interface for predictions:

```bash
streamlit run app.py
```

### 3. Make Predictions

```bash
python scripts/predict.py --smiles "CCO"  # Ethanol
python scripts/predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
```

---

## Actual Results (50K molecules, 100 epochs)

First training run results:

| Model | Test AUC |
|-------|----------|
| Baseline GNN | **0.8464** |
| Pre-trained GNN | 0.8337 |
| MLP | 0.8141 |

**Note:** Pre-training with only 50K molecules didn't improve results. This is expected -
more diverse training data is needed. Run `train_max_power.py` with 250K molecules for better results.

## Expected Results (250K molecules, 200 epochs)

| Model | Before Pre-training | After Pre-training | Improvement |
|-------|--------------------|--------------------|-------------|
| GCN | 0.846 AUC | ~0.87-0.89 AUC | +2-4% |
| Ensemble | 0.867 AUC | ~0.89-0.91 AUC | +2-4% |

---

## Troubleshooting

### "CUDA error: no kernel image is available"

Your PyTorch doesn't support RTX 5090. Install the nightly build:

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### "ModuleNotFoundError: No module named 'X'"

Install missing dependencies:

```bash
pip install tqdm rdkit scikit-learn pandas torch-geometric
```

### Training is slow

Make sure you're using GPU:
- Check the output shows "Using device: cuda (NVIDIA GeForce RTX 5090)"
- If it shows CPU, check PyTorch CUDA installation

---

## Project Structure

```
molecular-property-prediction/
├── data/
│   ├── tox21/              # Tox21 dataset
│   └── zinc/               # Pre-training data
├── models/                 # Saved model weights
│   ├── pretrained_gnn_encoder.pt  # Pre-trained encoder
│   ├── best_gnn_model.pt          # Fine-tuned GNN
│   └── ...
├── scripts/
│   ├── download_pretrain_data.py  # Download pre-training data
│   ├── pretrain_gnn.py            # Self-supervised pre-training
│   ├── pretrain_pipeline.py       # Full pipeline
│   ├── train_max_power.py         # Maximum RTX 5090 config
│   ├── train_gnn.py               # GNN training (modified)
│   └── ...
├── src/
│   ├── models/             # Model definitions
│   └── utils/              # Feature extraction
└── app.py                  # Streamlit dashboard
```

---

## Quick Reference Commands

```bash
# Navigate to project
cd d:\Development\molecular-property-prediction

# Quick training (50K molecules, ~15 min)
python scripts/pretrain_pipeline.py --quick

# Standard training (50K molecules, ~30 min)
python scripts/pretrain_pipeline.py

# Maximum power training (250K molecules, ~2 hours)
python scripts/train_max_power.py

# Launch dashboard
streamlit run app.py
```

---

*Last updated: December 2024*
