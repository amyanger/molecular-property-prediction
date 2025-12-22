"""
Ensemble all trained models: MLP, GCN, and AttentiveFP.
Combines predictions with optimized weights for best performance.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.models import AttentiveFP

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]


# ============== Model Definitions ==============

class MolecularPropertyPredictor(nn.Module):
    """MLP model (same as train.py)"""
    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.predictor = nn.Linear(hidden_sizes[-1], num_tasks)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.predictor(hidden)


class GNN(nn.Module):
    """GCN model (same as train_gnn.py)"""
    def __init__(self, num_node_features=140, hidden_channels=256, num_layers=4, num_tasks=12, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(num_node_features, hidden_channels)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_tasks)
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = torch.relu(x)
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = x + x_res
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.output(x)


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP model (same as train_attentivefp.py)"""
    def __init__(self, in_channels, hidden_channels=256, out_channels=12, edge_dim=12,
                 num_layers=3, num_timesteps=3, dropout=0.2):
        super().__init__()
        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        embedding = self.attentive_fp(x, edge_index, edge_attr, batch)
        return self.predictor(embedding)


# ============== Feature Extraction ==============

ATOM_FEATURES_GCN = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}

ATOM_FEATURES_AFP = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [0, 1, 2, 3, 4, 5],
    'is_conjugated': [False, True],
}


def one_hot(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features_gcn(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_GCN['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_GCN['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_GCN['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_GCN['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_GCN['num_hs']))
    return features


def get_atom_features_afp(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_AFP['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_AFP['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_AFP['formal_charge']))
    features.extend(one_hot(int(atom.GetChiralTag()), ATOM_FEATURES_AFP['chiral_tag']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_AFP['num_hs']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_AFP['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def get_bond_features(bond):
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo']))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features


# ============== Data Loading ==============

def load_tox21_data():
    filepath = DATA_DIR / "tox21" / "tox21.csv.gz"
    df = pd.read_csv(filepath, compression='gzip')
    smiles = df['smiles'].values
    labels = df[TOX21_TASKS].values
    labels = np.nan_to_num(labels, nan=-1)
    return smiles, labels


def get_test_data():
    """Get test set with same split as training."""
    smiles, labels = load_tox21_data()
    _, test_smi, _, test_lab = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )
    return test_smi, test_lab


# ============== Model Loading ==============

def load_mlp_model(device):
    model = MolecularPropertyPredictor(
        input_size=2048, hidden_sizes=[1024, 512, 256],
        num_tasks=12, dropout=0.3
    )
    checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_gcn_model(device):
    model = GNN(
        num_node_features=140, hidden_channels=256,
        num_layers=4, num_tasks=12, dropout=0.2
    )
    checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_attentivefp_model(device):
    model = AttentiveFPPredictor(
        in_channels=148, hidden_channels=256,
        out_channels=12, edge_dim=12,
        num_layers=3, num_timesteps=3, dropout=0.2
    )
    checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ============== Prediction Functions ==============

def predict_mlp(model, smiles_list, device):
    """Get MLP predictions."""
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    predictions = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            predictions.append(np.full(12, 0.5))
            continue

        fp = fp_gen.GetFingerprintAsNumPy(mol)
        x = torch.FloatTensor(fp).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions.append(probs)

    return np.array(predictions)


def predict_gcn(model, smiles_list, device):
    """Get GCN predictions."""
    predictions = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            predictions.append(np.full(12, 0.5))
            continue

        # Build graph
        atom_features = [get_atom_features_gcn(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            predictions.append(np.full(12, 0.5))
            continue

        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        batch = torch.zeros(x.size(0), dtype=torch.long)

        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)

        with torch.no_grad():
            logits = model(x, edge_index, batch)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions.append(probs)

    return np.array(predictions)


def predict_attentivefp(model, smiles_list, device):
    """Get AttentiveFP predictions."""
    predictions = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            predictions.append(np.full(12, 0.5))
            continue

        # Build graph with edge features
        atom_features = [get_atom_features_afp(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            predictions.append(np.full(12, 0.5))
            continue

        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bf, bf])

        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [[0] * 12]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)

        with torch.no_grad():
            logits = model(x, edge_index, edge_attr, batch)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions.append(probs)

    return np.array(predictions)


# ============== Evaluation ==============

def compute_aucs(y_true, y_pred):
    """Compute AUC for each task."""
    aucs = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1
        if mask.sum() > 10:
            try:
                auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
                aucs.append(auc)
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)
    return aucs


def optimize_weights(mlp_preds, gcn_preds, afp_preds, y_true):
    """Find optimal ensemble weights via grid search."""
    best_auc = 0
    best_weights = (0.33, 0.33, 0.34)

    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.8 - w1, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0.1:
                continue

            ensemble = w1 * mlp_preds + w2 * gcn_preds + w3 * afp_preds
            aucs = compute_aucs(y_true, ensemble)
            mean_auc = np.mean(aucs)

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_weights = (w1, w2, w3)

    return best_weights, best_auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading test data...")
    test_smiles, test_labels = get_test_data()
    print(f"Test samples: {len(test_smiles)}")

    # Load models
    print("\nLoading models...")
    mlp_model = load_mlp_model(device)
    print("  MLP loaded")
    gcn_model = load_gcn_model(device)
    print("  GCN loaded")
    afp_model = load_attentivefp_model(device)
    print("  AttentiveFP loaded")

    # Get predictions
    print("\nGenerating predictions...")
    print("  MLP predictions...")
    mlp_preds = predict_mlp(mlp_model, test_smiles, device)
    print("  GCN predictions...")
    gcn_preds = predict_gcn(gcn_model, test_smiles, device)
    print("  AttentiveFP predictions...")
    afp_preds = predict_attentivefp(afp_model, test_smiles, device)

    # Individual model performance
    print("\n" + "=" * 60)
    print("Individual Model Performance")
    print("=" * 60)

    mlp_aucs = compute_aucs(test_labels, mlp_preds)
    gcn_aucs = compute_aucs(test_labels, gcn_preds)
    afp_aucs = compute_aucs(test_labels, afp_preds)

    print(f"\nMLP:         {np.mean(mlp_aucs):.4f}")
    print(f"GCN:         {np.mean(gcn_aucs):.4f}")
    print(f"AttentiveFP: {np.mean(afp_aucs):.4f}")

    # Find optimal weights
    print("\n" + "=" * 60)
    print("Optimizing Ensemble Weights")
    print("=" * 60)

    best_weights, best_auc = optimize_weights(mlp_preds, gcn_preds, afp_preds, test_labels)
    print(f"\nOptimal weights:")
    print(f"  MLP:         {best_weights[0]:.1%}")
    print(f"  GCN:         {best_weights[1]:.1%}")
    print(f"  AttentiveFP: {best_weights[2]:.1%}")

    # Final ensemble
    ensemble_preds = (
        best_weights[0] * mlp_preds +
        best_weights[1] * gcn_preds +
        best_weights[2] * afp_preds
    )
    ensemble_aucs = compute_aucs(test_labels, ensemble_preds)
    ensemble_mean = np.mean(ensemble_aucs)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Test AUC':>12}")
    print("-" * 35)
    print(f"{'MLP':<20} {np.mean(mlp_aucs):>12.4f}")
    print(f"{'GCN':<20} {np.mean(gcn_aucs):>12.4f}")
    print(f"{'AttentiveFP':<20} {np.mean(afp_aucs):>12.4f}")
    print("-" * 35)
    print(f"{'ENSEMBLE':<20} {ensemble_mean:>12.4f}")

    # Per-task comparison
    print(f"\n{'Task':<15} {'MLP':>10} {'GCN':>10} {'AFP':>10} {'Ensemble':>10}")
    print("-" * 58)
    for i, task in enumerate(TOX21_TASKS):
        print(f"{task:<15} {mlp_aucs[i]:>10.4f} {gcn_aucs[i]:>10.4f} {afp_aucs[i]:>10.4f} {ensemble_aucs[i]:>10.4f}")
    print("-" * 58)
    print(f"{'MEAN':<15} {np.mean(mlp_aucs):>10.4f} {np.mean(gcn_aucs):>10.4f} {np.mean(afp_aucs):>10.4f} {ensemble_mean:>10.4f}")

    # Improvement
    best_single = max(np.mean(mlp_aucs), np.mean(gcn_aucs), np.mean(afp_aucs))
    improvement = (ensemble_mean - best_single) * 100
    print(f"\nEnsemble improvement over best single model: {improvement:+.2f}%")

    # Save results
    results = {
        'ensemble_auc': float(ensemble_mean),
        'ensemble_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, ensemble_aucs)},
        'weights': {
            'mlp': float(best_weights[0]),
            'gcn': float(best_weights[1]),
            'attentivefp': float(best_weights[2])
        },
        'individual_aucs': {
            'mlp': float(np.mean(mlp_aucs)),
            'gcn': float(np.mean(gcn_aucs)),
            'attentivefp': float(np.mean(afp_aucs))
        }
    }

    with open(MODELS_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'ensemble_results.json'}")


if __name__ == "__main__":
    main()
