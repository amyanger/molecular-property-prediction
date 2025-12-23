"""
Self-supervised pre-training for GNN molecular property prediction.

Pre-training Tasks:
1. Masked Atom Prediction - predict features of masked atoms
2. Context Prediction - predict if subgraphs belong to same molecule

This script pre-trains the GNN encoder on a large unlabeled molecular dataset
(ZINC or ChEMBL), then saves the encoder weights for fine-tuning on Tox21.

Usage:
    python scripts/pretrain_gnn.py
    python scripts/pretrain_gnn.py --epochs 100 --batch_size 256
    python scripts/pretrain_gnn.py --data_path data/zinc/zinc_pretrain.csv
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from src.constants import DATA_DIR, MODELS_DIR
from src.utils import get_atom_features_gcn


class PretrainGNNEncoder(nn.Module):
    """
    GNN Encoder for self-supervised pre-training.

    Architecture matches the GNN model in src/models/gcn.py
    so weights can be transferred directly.
    """

    def __init__(
        self,
        num_node_features=141,
        hidden_channels=256,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection (matches GNN.input_proj)
        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        # Graph convolution layers (matches GNN.convs)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass - returns node embeddings.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node (optional)

        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        # Initial projection
        x = self.input_proj(x)
        x = torch.relu(x)

        # Message passing with residual connections
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = x + x_res  # Residual connection

        return x

    def get_graph_embedding(self, x, edge_index, batch):
        """Get graph-level embedding (for contrastive learning)."""
        node_emb = self.forward(x, edge_index, batch)
        x_mean = global_mean_pool(node_emb, batch)
        x_max = global_max_pool(node_emb, batch)
        return torch.cat([x_mean, x_max], dim=1)


class PretrainModel(nn.Module):
    """
    Full pre-training model with encoder + pre-training heads.
    """

    def __init__(
        self,
        num_node_features=141,
        hidden_channels=256,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = PretrainGNNEncoder(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        )

        # Masked atom prediction head
        self.atom_pred_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_node_features)
        )

        # Contrastive learning projection head
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 128)
        )

    def forward_masked(self, x, edge_index, mask_indices):
        """Forward pass for masked atom prediction."""
        node_emb = self.encoder(x, edge_index)
        masked_emb = node_emb[mask_indices]
        predictions = self.atom_pred_head(masked_emb)
        return predictions

    def forward_contrastive(self, x, edge_index, batch):
        """Forward pass for contrastive learning."""
        graph_emb = self.encoder.get_graph_embedding(x, edge_index, batch)
        proj = self.proj_head(graph_emb)
        return torch.nn.functional.normalize(proj, dim=1)


class PretrainDataset(Dataset):
    """Dataset for pre-training from SMILES CSV."""

    def __init__(self, smiles_list):
        super().__init__()
        self.graphs = []

        print("Converting SMILES to graphs...")
        for smi in tqdm(smiles_list):
            graph = self._smiles_to_graph(smi)
            if graph is not None:
                self.graphs.append(graph)

        print(f"Valid graphs: {len(self.graphs)}/{len(smiles_list)}")

    def _smiles_to_graph(self, smiles):
        """Convert SMILES to PyG Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features_gcn(atom))

        if len(atom_features) == 0:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get edges
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        if len(edge_index) == 0:
            edge_index = [[0, 0]]  # Self-loop for single atom

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def mask_atoms(data, mask_ratio=0.15):
    """
    Mask random atoms for self-supervised learning.

    Strategy: Replace atom features with zeros (mask token).
    The model must predict original features from context.

    Args:
        data: PyG Data object
        mask_ratio: Fraction of atoms to mask

    Returns:
        masked_x: Node features with masked atoms zeroed
        mask_indices: Indices of masked atoms
        original_features: Original features for loss computation
    """
    num_atoms = data.x.size(0)
    num_mask = max(1, int(num_atoms * mask_ratio))

    # Random mask indices
    perm = torch.randperm(num_atoms)
    mask_indices = perm[:num_mask]

    # Store original features
    original_features = data.x[mask_indices].clone()

    # Create masked version
    masked_x = data.x.clone()
    masked_x[mask_indices] = 0  # Zero out masked atoms

    return masked_x, mask_indices, original_features


def augment_graph(data, drop_edge_ratio=0.1, drop_node_ratio=0.1):
    """
    Augment graph for contrastive learning.

    Args:
        data: PyG Data object
        drop_edge_ratio: Fraction of edges to drop
        drop_node_ratio: Fraction of nodes to drop

    Returns:
        Augmented Data object
    """
    x = data.x.clone()
    edge_index = data.edge_index.clone()

    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    # Drop edges
    if drop_edge_ratio > 0 and num_edges > 2:
        keep_edges = torch.rand(num_edges) > drop_edge_ratio
        edge_index = edge_index[:, keep_edges]

    # Add noise to features (feature masking)
    if drop_node_ratio > 0:
        mask = torch.rand(num_nodes, 1) > drop_node_ratio
        x = x * mask.float()

    return Data(x=x, edge_index=edge_index)


def contrastive_loss(z1, z2, temperature=0.5):
    """
    NT-Xent contrastive loss (SimCLR style).

    Args:
        z1, z2: Embeddings of two augmented views [batch_size, dim]
        temperature: Softmax temperature

    Returns:
        Contrastive loss
    """
    batch_size = z1.size(0)

    # Cosine similarity
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -float('inf'))

    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z.device)

    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss


def train_epoch(model, dataloader, optimizer, device, mask_ratio=0.15, use_contrastive=True):
    """Train for one epoch with masked atom prediction + contrastive learning."""
    model.train()
    total_loss = 0
    total_mask_loss = 0
    total_contrast_loss = 0

    criterion = nn.MSELoss()

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Task 1: Masked Atom Prediction
        masked_x, mask_indices, original_features = mask_atoms(batch)
        masked_x = masked_x.to(device)
        mask_indices = mask_indices.to(device)
        original_features = original_features.to(device)

        predictions = model.forward_masked(masked_x, batch.edge_index, mask_indices)
        mask_loss = criterion(predictions, original_features)

        # Task 2: Contrastive Learning (optional)
        if use_contrastive and batch.x.size(0) > 1:
            # Create two augmented views
            z1 = model.forward_contrastive(batch.x, batch.edge_index, batch.batch)

            # Augment and get second view
            aug_x = batch.x.clone()
            noise = torch.randn_like(aug_x) * 0.1
            aug_x = aug_x + noise
            z2 = model.forward_contrastive(aug_x, batch.edge_index, batch.batch)

            contrast_loss = contrastive_loss(z1, z2)
            loss = mask_loss + 0.1 * contrast_loss
            total_contrast_loss += contrast_loss.item()
        else:
            loss = mask_loss
            contrast_loss = torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mask_loss += mask_loss.item()

    n = len(dataloader)
    return {
        'total': total_loss / n,
        'mask': total_mask_loss / n,
        'contrast': total_contrast_loss / n if use_contrastive else 0
    }


def save_pretrained_weights(model, save_path, config):
    """
    Save pre-trained encoder weights in format compatible with GNN model.

    The encoder weights are saved with keys matching src/models/gcn.py:
    - input_proj.weight, input_proj.bias
    - convs.0.lin.weight, convs.0.bias, etc.
    - batch_norms.0.weight, batch_norms.0.bias, etc.
    """
    encoder_state = model.encoder.state_dict()

    torch.save({
        'encoder_state_dict': encoder_state,
        'config': config
    }, save_path)

    print(f"Pre-trained encoder saved to: {save_path}")


def main(args):
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("Using device: cpu (forced)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using device: cpu")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-training data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Default paths to try
        possible_paths = [
            DATA_DIR / "zinc" / "zinc_pretrain.csv",
            DATA_DIR / "chembl" / "chembl_pretrain.csv",
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            print("No pre-training data found. Please run first:")
            print("  python scripts/download_pretrain_data.py")
            return

    print(f"\nLoading pre-training data from: {data_path}")
    df = pd.read_csv(data_path)
    smiles_list = df['smiles'].tolist()

    # Limit if specified
    if args.max_molecules > 0:
        smiles_list = smiles_list[:args.max_molecules]

    print(f"Total molecules: {len(smiles_list)}")

    # Create dataset
    dataset = PretrainDataset(smiles_list)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )

    # Get feature dimension from first sample
    num_node_features = dataset[0].x.shape[1]
    print(f"Node features: {num_node_features}")

    # Create model
    model = PretrainModel(
        num_node_features=num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting pre-training for {args.epochs} epochs...")
    history = {'total_loss': [], 'mask_loss': [], 'contrast_loss': []}
    best_loss = float('inf')

    for epoch in range(args.epochs):
        losses = train_epoch(
            model, dataloader, optimizer, device,
            mask_ratio=args.mask_ratio,
            use_contrastive=args.use_contrastive
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['total_loss'].append(losses['total'])
        history['mask_loss'].append(losses['mask'])
        history['contrast_loss'].append(losses['contrast'])

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {losses['total']:.4f} (mask: {losses['mask']:.4f}, "
              f"contrast: {losses['contrast']:.4f}) | LR: {current_lr:.6f}")

        # Save best model
        if losses['total'] < best_loss:
            best_loss = losses['total']
            config = {
                'num_node_features': num_node_features,
                'hidden_channels': args.hidden_channels,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'pretrain_epochs': epoch + 1,
                'pretrain_loss': best_loss,
                'mask_ratio': args.mask_ratio,
                'use_contrastive': args.use_contrastive,
            }
            save_pretrained_weights(
                model,
                MODELS_DIR / 'pretrained_gnn_encoder.pt',
                config
            )

    # Save training history
    history_path = MODELS_DIR / 'pretrain_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Pre-training Complete!")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {MODELS_DIR / 'pretrained_gnn_encoder.pt'}")
    print(f"\nNext step: Fine-tune on Tox21 with:")
    print(f"  python scripts/train_gnn.py --pretrained")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train GNN on large molecular dataset')

    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to pre-training SMILES CSV')
    parser.add_argument('--max_molecules', type=int, default=0,
                        help='Max molecules to use (0 = all)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden channels (must match fine-tuning)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers (must match fine-tuning)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Pre-training task arguments
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Ratio of atoms to mask')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Use contrastive learning')
    parser.add_argument('--no_contrastive', action='store_false', dest='use_contrastive',
                        help='Disable contrastive learning')

    # Device
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')

    args = parser.parse_args()
    main(args)
