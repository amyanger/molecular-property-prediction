"""
Train Graph Neural Network for molecular property prediction.
Converts molecules to graphs and uses message passing for predictions.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from tqdm import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# Import from shared modules
from src.models import GNN
from src.constants import TOX21_TASKS, DATA_DIR, MODELS_DIR
from src.utils import get_atom_features_gcn as get_atom_features


def mol_to_graph(smiles, labels):
    """
    Convert a molecule (SMILES) to a PyG graph Data object.

    Args:
        smiles: SMILES string
        labels: numpy array of labels for all tasks

    Returns:
        torch_geometric.data.Data or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edges (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        # Single atom molecule - add self-loop
        edge_index = [[0, 0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Labels
    y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, y=y, smiles=smiles)


class MoleculeGraphDataset(Dataset):
    """PyTorch Geometric dataset for molecular graphs."""

    def __init__(self, smiles_list, labels_array):
        super().__init__()
        self.graphs = []

        print("Converting molecules to graphs...")
        for smi, lab in tqdm(zip(smiles_list, labels_array), total=len(smiles_list)):
            graph = mol_to_graph(smi, lab)
            if graph is not None:
                self.graphs.append(graph)

        print(f"Valid graphs: {len(self.graphs)}/{len(smiles_list)}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


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
        logits = model(batch.x, batch.edge_index, batch.batch)

        # Get labels and create mask
        y = batch.y.view(-1, 12)
        mask = (y != -1).float()

        # Masked loss
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
            logits = model(batch.x, batch.edge_index, batch.batch)
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
    # Setup device - use GPU if available
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

    # Split data
    train_smi, test_smi, train_lab, test_lab = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )
    train_smi, val_smi, train_lab, val_lab = train_test_split(
        train_smi, train_lab, test_size=0.125, random_state=42
    )

    print(f"Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}")

    # Create datasets
    print("\nCreating graph datasets...")
    train_dataset = MoleculeGraphDataset(train_smi, train_lab)
    val_dataset = MoleculeGraphDataset(val_smi, val_lab)
    test_dataset = MoleculeGraphDataset(test_smi, test_lab)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Calculate number of node features
    sample = train_dataset[0]
    num_node_features = sample.x.shape[1]
    print(f"\nNode features: {num_node_features}")

    # Create model
    model = GNN(
        num_node_features=num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_tasks=len(TOX21_TASKS),
        dropout=args.dropout,
        conv_type=args.conv_type
    ).to(device)

    print(f"Model: GNN ({args.conv_type.upper()})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    patience = 10
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

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'args': vars(args)
            }, MODELS_DIR / 'best_gnn_model.pt')
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

    # Load best model
    checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_aucs = evaluate(model, test_loader, device)
    test_auc = np.mean(test_aucs)

    print(f"\nOverall Test AUC-ROC: {test_auc:.4f}")
    print(f"\nPer-task AUC-ROC:")
    for task, auc in zip(TOX21_TASKS, test_aucs):
        print(f"  {task}: {auc:.4f}")

    # Compare with MLP
    try:
        with open(MODELS_DIR / 'results.json', 'r') as f:
            mlp_results = json.load(f)
        mlp_auc = mlp_results['test_auc_mean']
        print(f"\n{'='*60}")
        print("Comparison with MLP")
        print(f"{'='*60}")
        print(f"MLP AUC:  {mlp_auc:.4f}")
        print(f"GNN AUC:  {test_auc:.4f}")
        print(f"Improvement: {(test_auc - mlp_auc) * 100:+.2f}%")
    except FileNotFoundError:
        pass

    # Save results
    results = {
        'model': f'gnn_{args.conv_type}',
        'test_auc_mean': float(test_auc),
        'test_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, test_aucs)},
        'best_val_auc': float(best_val_auc),
        'epochs': epoch + 1,
        'history': history,
        'config': vars(args)
    }

    with open(MODELS_DIR / 'gnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'gnn_results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN for molecular property prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--conv_type', type=str, default='gcn', choices=['gcn', 'gat'],
                        help='Type of graph convolution')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()
    main(args)
