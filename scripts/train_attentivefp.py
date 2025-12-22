"""
Train AttentiveFP for molecular property prediction.
AttentiveFP uses graph attention with edge features - state-of-the-art for molecular tasks.
Expected AUC: ~0.85+ on Tox21
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from tqdm import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.loader import DataLoader

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

# Tox21 task names
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]

# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 118 elements
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

# Bond feature dimensions
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
    """One-hot encode a value."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features(atom):
    """
    Get atom features for AttentiveFP.

    Features (39 total):
    - Atomic number (one-hot, 118)
    - Degree (one-hot, 7)
    - Formal charge (one-hot, 6)
    - Chiral tag (one-hot, 4)
    - Num Hs (one-hot, 5)
    - Hybridization (one-hot, 6)
    - Is aromatic (1)
    - Is in ring (1)
    """
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def get_bond_features(bond):
    """
    Get bond/edge features for AttentiveFP.

    Features (12 total):
    - Bond type (one-hot, 4)
    - Stereo (one-hot, 6)
    - Is conjugated (1)
    - Is in ring (1)
    """
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo']))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features


def mol_to_graph_attentivefp(smiles, labels):
    """
    Convert molecule to graph with both node and edge features.
    Required format for AttentiveFP.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices and features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        # Single atom - add self-loop with dummy edge features
        edge_indices = [[0, 0]]
        edge_features = [[0] * 12]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Labels
    y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


class MoleculeDatasetAttentiveFP(Dataset):
    """Dataset for AttentiveFP with edge features."""

    def __init__(self, smiles_list, labels_array):
        super().__init__()
        self.graphs = []

        print("Converting molecules to graphs with edge features...")
        for smi, lab in tqdm(zip(smiles_list, labels_array), total=len(smiles_list)):
            graph = mol_to_graph_attentivefp(smi, lab)
            if graph is not None:
                self.graphs.append(graph)

        print(f"Valid graphs: {len(self.graphs)}/{len(smiles_list)}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


class AttentiveFPPredictor(nn.Module):
    """
    AttentiveFP wrapper for multi-task toxicity prediction.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        out_channels=12,
        edge_dim=12,
        num_layers=3,
        num_timesteps=3,
        dropout=0.2
    ):
        super().__init__()

        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Output embedding
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

        # Multi-task prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Get graph-level embedding from AttentiveFP
        embedding = self.attentive_fp(x, edge_index, edge_attr, batch)
        # Predict toxicity
        return self.predictor(embedding)


def load_tox21_data():
    """Load Tox21 dataset."""
    filepath = DATA_DIR / "tox21" / "tox21.csv.gz"
    df = pd.read_csv(filepath, compression='gzip')

    smiles = df['smiles'].values
    labels = df[TOX21_TASKS].values
    labels = np.nan_to_num(labels, nan=-1)

    return smiles, labels


def compute_metrics(y_true, y_pred, mask):
    """Compute AUC-ROC for each task."""
    aucs = []
    for i in range(y_true.shape[1]):
        valid_mask = mask[:, i] == 1
        if valid_mask.sum() > 10:
            try:
                auc = roc_auc_score(y_true[valid_mask, i], y_pred[valid_mask, i])
                aucs.append(auc)
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)
    return aucs


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y.view(-1, 12)
        mask = (y != -1).float()

        loss = criterion(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            y = batch.y.view(-1, 12).cpu().numpy()
            mask = (y != -1)

            all_preds.append(probs)
            all_labels.append(y)
            all_masks.append(mask)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_masks = np.vstack(all_masks)

    aucs = compute_metrics(all_labels, all_preds, all_masks)
    return aucs


def main(args):
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("Using device: cpu (forced)")
    elif torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = torch.device('cuda')
            print(f"Using device: cuda")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError:
            print("CUDA error, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using device: cpu")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading Tox21 dataset...")
    smiles, labels = load_tox21_data()
    print(f"Total samples: {len(smiles)}")

    # Split data (same split as other models)
    train_smi, test_smi, train_lab, test_lab = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )
    train_smi, val_smi, train_lab, val_lab = train_test_split(
        train_smi, train_lab, test_size=0.125, random_state=42
    )

    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    # Create datasets
    print("\nCreating graph datasets with edge features...")
    train_dataset = MoleculeDatasetAttentiveFP(train_smi, train_lab)
    val_dataset = MoleculeDatasetAttentiveFP(val_smi, val_lab)
    test_dataset = MoleculeDatasetAttentiveFP(test_smi, test_lab)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Get feature dimensions from sample
    sample = train_dataset[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1]
    print(f"\nNode features: {num_node_features}")
    print(f"Edge features: {num_edge_features}")

    # Create model
    model = AttentiveFPPredictor(
        in_channels=num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=len(TOX21_TASKS),
        edge_dim=num_edge_features,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel: AttentiveFP")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    patience = 15
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_aucs = evaluate(model, val_loader, device)
        val_auc = np.mean(val_aucs)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'args': vars(args)
            }, MODELS_DIR / 'best_attentivefp_model.pt')
            print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_aucs = evaluate(model, test_loader, device)
    test_auc = np.mean(test_aucs)

    print(f"\nOverall Test AUC-ROC: {test_auc:.4f}")
    print(f"\nPer-task AUC-ROC:")
    for task, auc in zip(TOX21_TASKS, test_aucs):
        print(f"  {task}: {auc:.4f}")

    # Compare with other models
    print(f"\n{'='*60}")
    print("Comparison with Other Models")
    print(f"{'='*60}")

    comparisons = [
        ('MLP', 'results.json', 'test_auc_mean'),
        ('GNN (GCN)', 'gnn_results.json', 'test_auc_mean'),
    ]

    for name, filename, key in comparisons:
        try:
            with open(MODELS_DIR / filename, 'r') as f:
                data = json.load(f)
            other_auc = data[key]
            diff = (test_auc - other_auc) * 100
            print(f"{name}: {other_auc:.4f} (AttentiveFP is {diff:+.2f}%)")
        except FileNotFoundError:
            pass

    print(f"AttentiveFP: {test_auc:.4f}")

    # Save results
    results = {
        'model': 'attentivefp',
        'test_auc_mean': float(test_auc),
        'test_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, test_aucs)},
        'best_val_auc': float(best_val_auc),
        'epochs': epoch + 1,
        'history': history,
        'config': vars(args)
    }

    with open(MODELS_DIR / 'attentivefp_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'attentivefp_results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AttentiveFP for molecular property prediction')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of AttentiveFP layers')
    parser.add_argument('--num_timesteps', type=int, default=3, help='Number of attention timesteps')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()
    main(args)
