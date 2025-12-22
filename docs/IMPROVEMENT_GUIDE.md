# Improving Your Molecular Toxicity Model

This guide walks you through implementing an ensemble approach to boost your model's AUC from ~0.81 to ~0.83-0.84.

## Overview

We'll implement three improvements:
1. **Multiple fingerprint types** - Add MACCS keys and RDKit descriptors
2. **XGBoost model** - A gradient boosting baseline
3. **Ensemble predictions** - Combine MLP + XGBoost for better accuracy

## Prerequisites

Install additional dependencies:

```bash
# Activate your virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install XGBoost
pip install xgboost
```

---

## Step 1: Create Enhanced Feature Extractor

Create a new file `scripts/features.py`:

```python
"""
Enhanced molecular feature extraction.
Combines multiple fingerprint types for better predictions.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors


# RDKit molecular descriptors to compute
DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
    'NumHeteroatoms', 'RingCount'
]


class MolecularFeaturizer:
    """
    Extract multiple types of molecular features.

    Features:
    - Morgan fingerprints (2048 bits) - circular fingerprints like ECFP4
    - MACCS keys (167 bits) - predefined structural keys
    - RDKit descriptors (10 values) - computed molecular properties

    Total: 2225 features
    """

    def __init__(self, morgan_size=2048, morgan_radius=2):
        self.morgan_size = morgan_size
        self.morgan_radius = morgan_radius
        self.morgan_gen = GetMorganGenerator(
            radius=morgan_radius,
            fpSize=morgan_size
        )
        self.desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            DESCRIPTOR_NAMES
        )

    def featurize(self, smiles: str) -> np.ndarray | None:
        """
        Convert SMILES to feature vector.

        Returns:
            numpy array of shape (2225,) or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Morgan fingerprints (2048 bits)
        morgan_fp = self.morgan_gen.GetFingerprintAsNumPy(mol)

        # 2. MACCS keys (167 bits)
        maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))

        # 3. RDKit descriptors (10 values)
        descriptors = np.array(self.desc_calc.CalcDescriptors(mol))
        # Handle any NaN/inf values
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate all features
        features = np.concatenate([morgan_fp, maccs_fp, descriptors])

        return features.astype(np.float32)

    def featurize_batch(self, smiles_list: list) -> tuple[np.ndarray, list]:
        """
        Featurize a batch of molecules.

        Returns:
            (features array, valid indices)
        """
        features = []
        valid_indices = []

        for i, smi in enumerate(smiles_list):
            feat = self.featurize(smi)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)

        return np.array(features), valid_indices

    @property
    def feature_size(self) -> int:
        """Total number of features."""
        return self.morgan_size + 167 + len(DESCRIPTOR_NAMES)  # 2048 + 167 + 10 = 2225


# Quick test
if __name__ == "__main__":
    featurizer = MolecularFeaturizer()

    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "invalid_smiles",
    ]

    for smi in test_smiles:
        feat = featurizer.featurize(smi)
        if feat is not None:
            print(f"SMILES: {smi[:30]}... -> {feat.shape} features")
        else:
            print(f"SMILES: {smi} -> Invalid")

    print(f"\nTotal feature size: {featurizer.feature_size}")
```

**Test it:**
```bash
python scripts/features.py
```

---

## Step 2: Create XGBoost Training Script

Create `scripts/train_xgboost.py`:

```python
"""
Train XGBoost model on Tox21 dataset.
Uses enhanced molecular features.
"""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle

from features import MolecularFeaturizer

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]


def load_tox21_data():
    """Load Tox21 dataset."""
    import pandas as pd
    filepath = DATA_DIR / "tox21" / "tox21.csv.gz"
    df = pd.read_csv(filepath, compression='gzip')

    smiles = df['smiles'].values
    labels = df[TOX21_TASKS].values
    labels = np.nan_to_num(labels, nan=-1)

    return smiles, labels


def train_xgboost_multitask(X_train, y_train, X_val, y_val):
    """
    Train one XGBoost model per task (multi-label classification).

    Returns:
        List of trained XGBoost models
    """
    models = []

    for task_idx, task_name in enumerate(tqdm(TOX21_TASKS, desc="Training XGBoost")):
        # Get labels for this task
        y_task_train = y_train[:, task_idx]
        y_task_val = y_val[:, task_idx]

        # Filter out missing labels (-1)
        train_mask = y_task_train != -1
        val_mask = y_task_val != -1

        X_task_train = X_train[train_mask]
        y_task_train = y_task_train[train_mask]
        X_task_val = X_val[val_mask]
        y_task_val = y_task_val[val_mask]

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20,
            verbosity=0
        )

        model.fit(
            X_task_train, y_task_train,
            eval_set=[(X_task_val, y_task_val)],
            verbose=False
        )

        models.append(model)

    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate XGBoost models on test set."""
    aucs = []

    for task_idx, (model, task_name) in enumerate(zip(models, TOX21_TASKS)):
        y_task = y_test[:, task_idx]
        mask = y_task != -1

        if mask.sum() > 10:
            y_pred = model.predict_proba(X_test[mask])[:, 1]
            auc = roc_auc_score(y_task[mask], y_pred)
            aucs.append(auc)
        else:
            aucs.append(0.5)

    return aucs


def main():
    print("Loading data...")
    smiles, labels = load_tox21_data()

    print("Computing enhanced features...")
    featurizer = MolecularFeaturizer()
    features, valid_indices = featurizer.featurize_batch(smiles)
    labels = labels[valid_indices]

    print(f"Feature shape: {features.shape}")

    # Split data (same split as MLP for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train XGBoost
    print("\nTraining XGBoost models (one per task)...")
    models = train_xgboost_multitask(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nEvaluating on test set...")
    test_aucs = evaluate_models(models, X_test, y_test)
    mean_auc = np.mean(test_aucs)

    print(f"\n{'='*50}")
    print(f"XGBoost Results")
    print(f"{'='*50}")
    print(f"\nOverall Test AUC: {mean_auc:.4f}")
    print(f"\nPer-task AUC:")
    for task, auc in zip(TOX21_TASKS, test_aucs):
        print(f"  {task}: {auc:.4f}")

    # Save models
    with open(MODELS_DIR / 'xgboost_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"\nModels saved to {MODELS_DIR / 'xgboost_models.pkl'}")

    # Save results
    results = {
        'model': 'xgboost',
        'test_auc_mean': float(mean_auc),
        'test_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, test_aucs)},
        'feature_size': featurizer.feature_size
    }
    with open(MODELS_DIR / 'xgboost_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

**Train XGBoost:**
```bash
python scripts/train_xgboost.py
```

Expected output: XGBoost should achieve ~0.79-0.82 AUC on its own.

---

## Step 3: Create Ensemble Predictor

Create `scripts/ensemble_predict.py`:

```python
"""
Ensemble predictor combining MLP and XGBoost.
Averages predictions for improved accuracy.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from rdkit import Chem

from features import MolecularFeaturizer
from train import MolecularPropertyPredictor

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]


class EnsemblePredictor:
    """
    Ensemble of MLP and XGBoost models.

    Combines predictions with optional weighting.
    Also provides confidence scores.
    """

    def __init__(self, mlp_weight=0.5, xgb_weight=0.5):
        self.mlp_weight = mlp_weight
        self.xgb_weight = xgb_weight

        # Feature extractors
        self.featurizer = MolecularFeaturizer()

        # Load MLP model
        self.mlp_model = self._load_mlp()

        # Load XGBoost models
        self.xgb_models = self._load_xgboost()

    def _load_mlp(self):
        """Load trained MLP model."""
        model = MolecularPropertyPredictor(
            input_size=2048,
            hidden_sizes=[1024, 512, 256],
            num_tasks=12,
            dropout=0.3
        )
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _load_xgboost(self):
        """Load trained XGBoost models."""
        with open(MODELS_DIR / 'xgboost_models.pkl', 'rb') as f:
            return pickle.load(f)

    def predict(self, smiles: str) -> dict | None:
        """
        Predict toxicity with ensemble.

        Returns:
            dict with 'probabilities', 'confidence', 'mlp_preds', 'xgb_preds'
            or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get features
        full_features = self.featurizer.featurize(smiles)
        morgan_features = full_features[:2048]  # MLP only uses Morgan

        # MLP predictions
        with torch.no_grad():
            x = torch.FloatTensor(morgan_features).unsqueeze(0)
            logits = self.mlp_model(x)
            mlp_probs = torch.sigmoid(logits).numpy()[0]

        # XGBoost predictions
        xgb_probs = np.array([
            model.predict_proba(full_features.reshape(1, -1))[0, 1]
            for model in self.xgb_models
        ])

        # Ensemble (weighted average)
        ensemble_probs = (
            self.mlp_weight * mlp_probs +
            self.xgb_weight * xgb_probs
        )

        # Confidence: how much the models agree (1 - variance)
        # High confidence = models agree, low confidence = models disagree
        disagreement = np.abs(mlp_probs - xgb_probs)
        confidence = 1 - disagreement

        return {
            'probabilities': ensemble_probs,
            'confidence': confidence,
            'mlp_predictions': mlp_probs,
            'xgb_predictions': xgb_probs
        }

    def predict_batch(self, smiles_list: list) -> list:
        """Predict for multiple molecules."""
        return [self.predict(smi) for smi in smiles_list]


def print_prediction(name: str, smiles: str, result: dict):
    """Pretty print ensemble prediction."""
    print(f"\n{'='*70}")
    print(f"Molecule: {name}")
    print(f"SMILES: {smiles}")
    print(f"{'='*70}")

    probs = result['probabilities']
    conf = result['confidence']
    mlp = result['mlp_predictions']
    xgb = result['xgb_predictions']

    avg_tox = np.mean(probs)
    avg_conf = np.mean(conf)

    risk = "HIGH" if avg_tox > 0.5 else "MODERATE" if avg_tox > 0.3 else "LOW"

    print(f"\nOverall Toxicity: {avg_tox:.1%} ({risk} RISK)")
    print(f"Average Confidence: {avg_conf:.1%}")

    print(f"\n{'Task':<15} {'Ensemble':>10} {'MLP':>10} {'XGBoost':>10} {'Confidence':>12}")
    print("-" * 60)

    for i, task in enumerate(TOX21_TASKS):
        conf_indicator = "***" if conf[i] > 0.9 else "**" if conf[i] > 0.7 else "*" if conf[i] > 0.5 else ""
        print(f"{task:<15} {probs[i]:>10.1%} {mlp[i]:>10.1%} {xgb[i]:>10.1%} {conf[i]:>10.1%} {conf_indicator}")


def main():
    print("Loading ensemble models...")
    ensemble = EnsemblePredictor(mlp_weight=0.5, xgb_weight=0.5)
    print("Models loaded!\n")

    # Test molecules
    test_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "DDT (pesticide)": "ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2",
    }

    for name, smiles in test_molecules.items():
        result = ensemble.predict(smiles)
        if result:
            print_prediction(name, smiles, result)

    # Interactive mode
    print("\n" + "="*70)
    print("Enter SMILES to predict (or 'quit' to exit)")

    while True:
        try:
            smiles = input("\nSMILES: ").strip()
            if smiles.lower() in ['quit', 'q', 'exit']:
                break

            result = ensemble.predict(smiles)
            if result:
                print_prediction("Custom", smiles, result)
            else:
                print("Invalid SMILES")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
```

---

## Step 4: Evaluate the Ensemble

Create `scripts/evaluate_ensemble.py`:

```python
"""
Evaluate ensemble performance vs individual models.
"""

import json
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from features import MolecularFeaturizer
from train import MolecularPropertyPredictor

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]


def load_data_and_features():
    """Load test data with features."""
    # Load raw data
    filepath = DATA_DIR / "tox21" / "tox21.csv.gz"
    df = pd.read_csv(filepath, compression='gzip')
    smiles = df['smiles'].values
    labels = df[TOX21_TASKS].values
    labels = np.nan_to_num(labels, nan=-1)

    # Compute features
    featurizer = MolecularFeaturizer()
    features, valid_indices = featurizer.featurize_batch(smiles)
    labels = labels[valid_indices]
    smiles = smiles[valid_indices]

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    _, smiles_test, _, _ = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )

    return X_test, y_test, smiles_test


def get_mlp_predictions(smiles_list, features):
    """Get MLP predictions (uses Morgan fingerprints only)."""
    model = MolecularPropertyPredictor(
        input_size=2048,
        hidden_sizes=[1024, 512, 256],
        num_tasks=12,
        dropout=0.3
    )
    checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # MLP uses only Morgan fingerprints (first 2048 features)
    morgan_features = features[:, :2048]

    with torch.no_grad():
        x = torch.FloatTensor(morgan_features)
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()

    return probs


def get_xgboost_predictions(features):
    """Get XGBoost predictions."""
    with open(MODELS_DIR / 'xgboost_models.pkl', 'rb') as f:
        models = pickle.load(f)

    predictions = []
    for model in models:
        pred = model.predict_proba(features)[:, 1]
        predictions.append(pred)

    return np.array(predictions).T  # Shape: (n_samples, n_tasks)


def compute_aucs(y_true, y_pred):
    """Compute AUC for each task."""
    aucs = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1
        if mask.sum() > 10:
            auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
            aucs.append(auc)
        else:
            aucs.append(0.5)
    return aucs


def main():
    print("Loading test data...")
    X_test, y_test, smiles_test = load_data_and_features()
    print(f"Test samples: {len(X_test)}")

    print("\nGetting MLP predictions...")
    mlp_preds = get_mlp_predictions(smiles_test, X_test)

    print("Getting XGBoost predictions...")
    xgb_preds = get_xgboost_predictions(X_test)

    # Ensemble with different weights
    print("\nEvaluating ensemble with different weights...")

    weights_to_try = [
        (1.0, 0.0, "MLP only"),
        (0.0, 1.0, "XGBoost only"),
        (0.5, 0.5, "Equal weight"),
        (0.6, 0.4, "MLP 60%, XGB 40%"),
        (0.7, 0.3, "MLP 70%, XGB 30%"),
    ]

    results = []

    for mlp_w, xgb_w, name in weights_to_try:
        ensemble_preds = mlp_w * mlp_preds + xgb_w * xgb_preds
        aucs = compute_aucs(y_test, ensemble_preds)
        mean_auc = np.mean(aucs)
        results.append((name, mean_auc, aucs))
        print(f"  {name}: {mean_auc:.4f}")

    # Find best
    best = max(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best[0]} with AUC {best[1]:.4f}")

    # Detailed comparison
    print(f"\n{'='*70}")
    print("Per-Task Comparison")
    print(f"{'='*70}")
    print(f"\n{'Task':<15} {'MLP':>10} {'XGBoost':>10} {'Ensemble':>10} {'Best':>10}")
    print("-" * 55)

    mlp_aucs = compute_aucs(y_test, mlp_preds)
    xgb_aucs = compute_aucs(y_test, xgb_preds)
    ensemble_aucs = best[2]

    for i, task in enumerate(TOX21_TASKS):
        best_model = "MLP" if mlp_aucs[i] > xgb_aucs[i] else "XGB"
        if ensemble_aucs[i] >= max(mlp_aucs[i], xgb_aucs[i]):
            best_model = "ENS"
        print(f"{task:<15} {mlp_aucs[i]:>10.4f} {xgb_aucs[i]:>10.4f} {ensemble_aucs[i]:>10.4f} {best_model:>10}")

    print("-" * 55)
    print(f"{'MEAN':<15} {np.mean(mlp_aucs):>10.4f} {np.mean(xgb_aucs):>10.4f} {np.mean(ensemble_aucs):>10.4f}")

    # Save results
    ensemble_results = {
        'best_config': best[0],
        'test_auc_mean': float(best[1]),
        'test_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, best[2])},
        'comparison': {
            'mlp_mean': float(np.mean(mlp_aucs)),
            'xgboost_mean': float(np.mean(xgb_aucs)),
            'ensemble_mean': float(best[1])
        }
    }

    with open(MODELS_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'ensemble_results.json'}")


if __name__ == "__main__":
    main()
```

---

## Running the Full Pipeline

Execute these commands in order:

```bash
# 1. Make sure you're in the project directory with venv activated
cd molecular-property-prediction
.\venv\Scripts\activate  # Windows

# 2. Install XGBoost if not already installed
pip install xgboost

# 3. Test the feature extractor
python scripts/features.py

# 4. Train XGBoost (takes a few minutes)
python scripts/train_xgboost.py

# 5. Evaluate the ensemble
python scripts/evaluate_ensemble.py

# 6. Run interactive predictions
python scripts/ensemble_predict.py
```

---

## Expected Results

| Model | Expected AUC |
|-------|--------------|
| MLP (current) | ~0.810 |
| XGBoost | ~0.790-0.820 |
| **Ensemble** | **~0.830-0.840** |

The ensemble typically outperforms individual models because:
- MLP and XGBoost make different types of errors
- Averaging reduces variance
- Enhanced features help XGBoost capture patterns MLP misses

---

## Next Steps After This

Once you've implemented the ensemble, consider:

1. **Hyperparameter tuning** - Use Optuna to optimize XGBoost and ensemble weights
2. **More fingerprints** - Add AtomPair, TopologicalTorsion fingerprints
3. **Graph Neural Networks** - PyTorch Geometric for molecular graphs (biggest potential gain)

---

## Troubleshooting

**ImportError: No module named 'xgboost'**
```bash
pip install xgboost
```

**CUDA out of memory**
- The XGBoost training uses CPU by default, so this shouldn't happen
- If MLP training fails, add `--cpu` flag

**Low XGBoost performance**
- Try increasing `n_estimators` to 500
- Try `max_depth=8`
- Add more features (more fingerprint types)

**Ensemble not improving**
- Try different weight combinations
- Ensure you're using the same train/test split (random_state=42)
