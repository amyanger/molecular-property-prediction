"""
Train molecular property prediction models on Tox21 dataset.
Uses PyTorch with molecular fingerprints from RDKit.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm
import json

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


class MoleculeDataset(Dataset):
    """PyTorch dataset for molecular property prediction."""

    def __init__(self, smiles_list, labels, fingerprint_size=2048):
        self.smiles_list = smiles_list
        self.labels = labels
        self.fingerprint_size = fingerprint_size
        self.valid_indices = []
        self.fingerprints = []

        # Pre-compute fingerprints using new RDKit API
        print("Computing molecular fingerprints...")
        fp_gen = GetMorganGenerator(radius=2, fpSize=fingerprint_size)
        for i, smi in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Morgan fingerprint (ECFP4-like)
                fp = fp_gen.GetFingerprintAsNumPy(mol)
                self.fingerprints.append(fp)
                self.valid_indices.append(i)

        self.fingerprints = np.array(self.fingerprints)
        self.labels = labels[self.valid_indices]
        print(f"Valid molecules: {len(self.valid_indices)}/{len(smiles_list)}")

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.fingerprints[idx])
        y = torch.FloatTensor(self.labels[idx])
        return x, y


class MolecularPropertyPredictor(nn.Module):
    """
    Neural network for predicting molecular properties.
    Architecture: Fingerprint -> MLP with residual connections -> Multi-task output
    """

    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers with batch norm and dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.encoder = nn.Sequential(*layers)

        # Multi-task prediction head (one output per toxicity endpoint)
        self.predictor = nn.Linear(hidden_sizes[-1], num_tasks)

    def forward(self, x):
        hidden = self.encoder(x)
        logits = self.predictor(hidden)
        return logits


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for molecular properties.
    Treats fingerprint bits as a sequence of features.
    """

    def __init__(self, input_size=2048, d_model=256, nhead=8, num_layers=4, num_tasks=12, dropout=0.1):
        super().__init__()

        # Project fingerprint to embedding space
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_tasks)
        )

    def forward(self, x):
        # x: (batch, fingerprint_size)
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, d_model)
        return self.output(x)


def load_tox21_data():
    """Load Tox21 dataset."""
    filepath = DATA_DIR / "tox21" / "tox21.csv.gz"
    df = pd.read_csv(filepath, compression='gzip')

    smiles = df['smiles'].values
    labels = df[TOX21_TASKS].values

    # Replace NaN with -1 (will be masked during training)
    labels = np.nan_to_num(labels, nan=-1)

    return smiles, labels


def compute_metrics(y_true, y_pred, mask):
    """Compute AUC-ROC for each task, ignoring masked values."""
    aucs = []
    for i in range(y_true.shape[1]):
        valid_mask = mask[:, i] == 1
        if valid_mask.sum() > 10:  # Need enough samples
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

    for batch_x, batch_y in tqdm(dataloader, desc="Training"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Create mask for valid labels (not -1)
        mask = (batch_y != -1).float()

        optimizer.zero_grad()
        logits = model(batch_x)

        # Masked loss - only compute for valid labels
        loss = criterion(logits, batch_y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()

            mask = (batch_y != -1).numpy()

            all_preds.append(probs)
            all_labels.append(batch_y.numpy())
            all_masks.append(mask)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_masks = np.vstack(all_masks)

    aucs = compute_metrics(all_labels, all_preds, all_masks)
    return aucs


def main(args):
    # Setup device
    # Note: RTX 5090 (Blackwell sm_120) requires PyTorch nightly for CUDA support
    if args.cpu:
        device = torch.device('cpu')
        print("Using device: cpu (forced via --cpu flag)")
    elif torch.cuda.is_available():
        try:
            # Test if CUDA actually works with current GPU
            torch.zeros(1).cuda()
            device = torch.device('cuda')
            print(f"Using device: cuda")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"CUDA available but kernel not supported (new GPU arch like Blackwell?)")
            print("Falling back to CPU. Install PyTorch nightly for RTX 5090 support.")
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
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_smiles, train_labels, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
    )

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    # Create datasets
    train_dataset = MoleculeDataset(train_smiles, train_labels, args.fingerprint_size)
    val_dataset = MoleculeDataset(val_smiles, val_labels, args.fingerprint_size)
    test_dataset = MoleculeDataset(test_smiles, test_labels, args.fingerprint_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    if args.model == 'mlp':
        model = MolecularPropertyPredictor(
            input_size=args.fingerprint_size,
            hidden_sizes=[1024, 512, 256],
            num_tasks=len(TOX21_TASKS),
            dropout=args.dropout
        )
    else:  # transformer
        model = TransformerPredictor(
            input_size=args.fingerprint_size,
            d_model=256,
            nhead=8,
            num_layers=4,
            num_tasks=len(TOX21_TASKS),
            dropout=args.dropout
        )

    model = model.to(device)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': [], 'test_auc': []}

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, MODELS_DIR / 'best_model.pt')
            print(f"  -> New best model saved!")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_aucs = evaluate(model, test_loader, device)
    test_auc = np.mean(test_aucs)

    print(f"\nOverall Test AUC-ROC: {test_auc:.4f}")
    print(f"\nPer-task AUC-ROC:")
    for task, auc in zip(TOX21_TASKS, test_aucs):
        print(f"  {task}: {auc:.4f}")

    # Save results
    results = {
        'model': args.model,
        'test_auc_mean': float(test_auc),
        'test_aucs': {task: float(auc) for task, auc in zip(TOX21_TASKS, test_aucs)},
        'best_val_auc': float(best_val_auc),
        'epochs': args.epochs,
        'history': history
    }

    with open(MODELS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {MODELS_DIR / 'results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train molecular property predictor')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--fingerprint_size', type=int, default=2048, help='Morgan fingerprint size')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()
    main(args)
