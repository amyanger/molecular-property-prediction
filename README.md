# Molecular Property Prediction

Deep learning models for predicting molecular toxicity across 12 biological endpoints. Trained on the Tox21 benchmark dataset used in pharmaceutical drug discovery.

## What This Project Does (Simple Terms)

**In one sentence:** Given any molecule, the model predicts whether it's toxic to humans across 12 different biological pathways.

### Real-World Example

You give it **Aspirin**:
```
CC(=O)OC1=CC=CC=C1C(=O)O
```

It tells you:
- Overall toxicity: **1.6% (LOW RISK)**
- Safe for hormone receptors ✓
- Safe for mitochondria ✓
- Safe for DNA damage ✓

You give it **DDT (pesticide)**:
```
ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2
```

It tells you:
- Overall toxicity: **80%+ (HIGH RISK)**
- Dangerous for hormone disruption ⚠
- Dangerous for cellular stress ⚠

### How It Works

```
SMILES String          →    Neural Network    →    Toxicity Predictions
(molecule formula)          (trained on           (12 scores, 0-100%)
                             7,831 molecules)
```

**Step 1: Molecule Input**
- Molecules are written as text strings called SMILES
- Example: `CCO` = ethanol (alcohol)

**Step 2: Convert to Numbers**
- **MLP**: Converts molecule to a 2048-bit fingerprint (like a barcode)
- **GNN**: Converts molecule to a graph (atoms = dots, bonds = lines connecting them)

**Step 3: Neural Network Predicts**
- Network learned patterns from 7,831 known toxic/safe molecules
- Outputs probability (0-100%) for each of 12 toxicity types

**Step 4: Ensemble**
- Combines 3 different models (MLP, GCN, AttentiveFP)
- Each model "votes" and the average is the final prediction

### The 12 Things It Checks

| Category | What It Detects |
|----------|-----------------|
| Hormone Disruption | Messes with estrogen/testosterone |
| Cancer Risk | Damages tumor suppressor genes |
| Cellular Stress | Harms mitochondria, causes oxidative stress |
| DNA Damage | Breaks or mutates DNA |

### Who Would Use This?

- **Pharma companies**: Screen drug candidates before expensive lab tests
- **Chemical manufacturers**: Check if new chemicals are safe
- **Researchers**: Prioritize which molecules to study

---

## Results

### Model Performance (Tox21 Benchmark)

| Model | Test AUC-ROC | Notes |
|-------|-------------|-------|
| MLP | 0.814 | Fingerprint-based baseline |
| GCN | 0.852 | Graph Neural Network |
| AttentiveFP | 0.852 | Attention-based GNN |
| **Ensemble** | **0.867** | Combined model (state-of-the-art) |
| Random Forest | ~0.750 | Traditional ML baseline |
| Published SOTA | ~0.850 | Literature benchmark |

### Why 3 Models?

| Model | How It Sees Molecules | Strength |
|-------|----------------------|----------|
| **MLP** | As a fingerprint (barcode) | Fast, simple |
| **GCN** | As a graph (atoms + bonds) | Understands structure |
| **AttentiveFP** | Graph + attention (focuses on important parts) | Most sophisticated |

Combining them (ensemble) = **0.867 AUC** (beats state-of-the-art accuracy)

### Per-Task Performance (Ensemble)

| Endpoint | AUC-ROC | Description |
|----------|---------|-------------|
| SR-MMP | 0.905 | Mitochondrial toxicity |
| NR-AhR | 0.904 | Dioxin-like toxicity |
| NR-AR-LBD | 0.890 | Androgen receptor binding |
| SR-p53 | 0.889 | Cancer risk (tumor suppressor) |
| SR-ATAD5 | 0.880 | DNA damage response |
| NR-ER-LBD | 0.869 | Estrogen receptor binding |

---

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
pip install rdkit pandas scikit-learn matplotlib tqdm torch-geometric streamlit plotly

# Download datasets
python scripts/download_data.py

# Train models
python scripts/train.py --model mlp --epochs 50      # MLP
python scripts/train_gnn.py --epochs 50              # GCN
python scripts/train_attentivefp.py --epochs 100     # AttentiveFP

# Run ensemble evaluation
python scripts/ensemble_all.py

# Run predictions
python scripts/predict.py

# Launch dashboard
streamlit run app.py
```

---

## Interactive Dashboard

The Streamlit dashboard lets you visualize results and make predictions interactively.

### Running the Dashboard

**Option 1: From the project directory**
```bash
cd d:\Development\molecular-property-prediction
.\venv\Scripts\activate
streamlit run app.py
```

**Option 2: From any terminal location (e.g., root of a drive)**
```powershell
# Windows PowerShell - run from anywhere
cd d:\Development\molecular-property-prediction; .\venv\Scripts\activate; streamlit run app.py
```

```bash
# Or as separate commands:
cd d:\Development\molecular-property-prediction
.\venv\Scripts\activate
streamlit run app.py
```

**Option 3: One-liner for Windows Command Prompt**
```cmd
cd /d d:\Development\molecular-property-prediction && venv\Scripts\activate && streamlit run app.py
```

After running, the dashboard will open automatically in your browser at `http://localhost:8501`

### Dashboard Features

- **Overview**: Project stats and model comparison charts with explanations
- **Predict Toxicity**: Enter any molecule (SMILES string) and get instant toxicity predictions
- **Model Comparison**: Per-task performance across all models
- **Training History**: Loss curves and validation metrics

Each page includes layman-friendly explanations so anyone can understand what the models are doing.

---

## Project Structure

```
molecular-property-prediction/
├── app.py                        # Streamlit dashboard
├── scripts/
│   ├── train.py                  # MLP training
│   ├── train_gnn.py              # GCN training
│   ├── train_attentivefp.py      # AttentiveFP training
│   ├── ensemble_all.py           # Ensemble evaluation
│   ├── predict.py                # Inference on new molecules
│   ├── compare_models.py         # Model comparison visualization
│   ├── download_data.py          # Dataset downloader
│   └── visualize_results.py      # Performance visualization
├── models/
│   ├── results.json              # MLP results
│   ├── gnn_results.json          # GCN results
│   ├── attentivefp_results.json  # AttentiveFP results
│   └── ensemble_results.json     # Ensemble results
├── data/                         # Datasets (downloaded separately)
├── docs/
│   └── IMPROVEMENT_GUIDE.md      # Guide for further improvements
└── notebooks/                    # Exploration notebooks
```

---

## Technical Details

### Model Architectures

**MLP Model:**
- Input: 2048-bit Morgan fingerprints (ECFP4)
- Hidden layers: 1024 → 512 → 256
- Multi-task output: 12 toxicity endpoints
- Dropout: 0.3, BatchNorm, AdamW optimizer

**GCN Model:**
- Input: Molecular graph (atoms as nodes, bonds as edges)
- 141-dimensional node features (atomic number, degree, charge, hybridization, aromaticity, ring membership)
- 4 GCN layers with residual connections
- Global mean + max pooling
- Hidden size: 256

**AttentiveFP Model:**
- Graph attention with edge features
- 148-dimensional node features + 12-dimensional edge features
- 3 attention layers, 3 timesteps
- Learns which atoms/bonds are most important

**Ensemble:**
- Optimized weights: MLP (10%) + GCN (50%) + AttentiveFP (40%)
- Grid search for optimal combination

### Dataset

**Tox21** - NIH toxicity screening data
- 7,831 compounds
- 12 biological assay endpoints
- Multi-label classification

---

## Hardware

Trained on NVIDIA RTX 5090 (32GB VRAM) with PyTorch 2.11 nightly + CUDA 12.8

---

## References

- [Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/)
- [MoleculeNet Benchmark](https://moleculenet.org/)
- [DeepChem](https://deepchem.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [AttentiveFP Paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)

---

## License

MIT License
