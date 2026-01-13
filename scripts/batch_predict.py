#!/usr/bin/env python
"""
Batch prediction script for processing multiple molecules.

Usage:
    python scripts/batch_predict.py --input molecules.csv --output predictions.csv
    python scripts/batch_predict.py --input molecules.txt --output predictions.json
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.models import MolecularPropertyPredictor, GNN, AttentiveFPPredictor
from src.constants import TOX21_TASKS, MODELS_DIR, DEFAULT_ENSEMBLE_WEIGHTS
from src.utils.features import get_atom_features_gcn, get_atom_features_afp, get_bond_features
from src.utils.smiles import validate_smiles


def load_models(device):
    """Load all trained models."""
    models = {}

    try:
        mlp = MolecularPropertyPredictor(input_size=2048, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', map_location=device, weights_only=True)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device).eval()
        models['mlp'] = mlp
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")

    try:
        gnn = GNN(num_node_features=141, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', map_location=device, weights_only=True)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        gnn.to(device).eval()
        models['gnn'] = gnn
    except Exception as e:
        print(f"Warning: Could not load GNN model: {e}")

    try:
        afp = AttentiveFPPredictor(in_channels=148, out_channels=12, edge_dim=12)
        checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', map_location=device, weights_only=True)
        afp.load_state_dict(checkpoint['model_state_dict'])
        afp.to(device).eval()
        models['attentivefp'] = afp
    except Exception as e:
        print(f"Warning: Could not load AttentiveFP model: {e}")

    return models


def smiles_to_fingerprint(smiles):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    return fp_gen.GetFingerprintAsNumPy(mol)


def smiles_to_graph(smiles):
    """Convert SMILES to graph for GNN."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [get_atom_features_gcn(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if not edge_index:
        edge_index = [[0, 0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, batch


def smiles_to_graph_afp(smiles):
    """Convert SMILES to graph with edge features for AttentiveFP."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [get_atom_features_afp(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_features.extend([feat, feat])

    if not edge_index:
        edge_index = [[0, 0]]
        edge_features = [[0] * 12]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, edge_attr, batch


def predict_single(smiles, models, device):
    """Get ensemble prediction for a single molecule."""
    is_valid, _ = validate_smiles(smiles)
    if not is_valid:
        return None

    predictions = []
    weights = []

    # MLP prediction
    if 'mlp' in models:
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            with torch.no_grad():
                x = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(device)
                probs = torch.sigmoid(models['mlp'](x)).cpu().numpy()[0]
                predictions.append(probs)
                weights.append(DEFAULT_ENSEMBLE_WEIGHTS['mlp'])

    # GNN prediction
    if 'gnn' in models:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            x, edge_index, batch = graph
            with torch.no_grad():
                x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
                probs = torch.sigmoid(models['gnn'](x, edge_index, batch)).cpu().numpy()[0]
                predictions.append(probs)
                weights.append(DEFAULT_ENSEMBLE_WEIGHTS['gcn'])

    # AttentiveFP prediction
    if 'attentivefp' in models:
        graph = smiles_to_graph_afp(smiles)
        if graph is not None:
            x, edge_index, edge_attr, batch = graph
            with torch.no_grad():
                x = x.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                batch = batch.to(device)
                probs = torch.sigmoid(models['attentivefp'](x, edge_index, edge_attr, batch)).cpu().numpy()[0]
                predictions.append(probs)
                weights.append(DEFAULT_ENSEMBLE_WEIGHTS['attentivefp'])

    if not predictions:
        return None

    # Weighted ensemble
    weights = np.array(weights) / sum(weights)
    return sum(w * p for w, p in zip(weights, predictions))


def load_input_file(filepath):
    """Load SMILES from input file (CSV or TXT)."""
    filepath = Path(filepath)

    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        # Look for SMILES column
        smiles_col = None
        for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'mol']:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            # Use first column
            smiles_col = df.columns[0]

        smiles_list = df[smiles_col].tolist()

        # Get ID column if present
        id_col = None
        for col in ['id', 'ID', 'name', 'Name', 'compound_id']:
            if col in df.columns:
                id_col = col
                break

        ids = df[id_col].tolist() if id_col else list(range(len(smiles_list)))

    else:  # TXT file - one SMILES per line
        with open(filepath, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        ids = list(range(len(smiles_list)))

    return ids, smiles_list


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    models = load_models(device)
    print(f"Loaded models: {list(models.keys())}")

    if not models:
        print("Error: No models available")
        sys.exit(1)

    # Load input
    print(f"Loading input from: {args.input}")
    ids, smiles_list = load_input_file(args.input)
    print(f"Loaded {len(smiles_list)} molecules")

    # Make predictions
    print("Making predictions...")
    results = []

    for idx, smiles in tqdm(zip(ids, smiles_list), total=len(smiles_list)):
        pred = predict_single(smiles, models, device)

        result = {
            'id': idx,
            'smiles': smiles,
            'valid': pred is not None,
        }

        if pred is not None:
            result['overall_score'] = float(np.mean(pred))
            for i, task in enumerate(TOX21_TASKS):
                result[task] = float(pred[i])
        else:
            result['overall_score'] = None
            for task in TOX21_TASKS:
                result[task] = None

        results.append(result)

    # Save output
    output_path = Path(args.output)

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:  # CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\nResults saved to: {output_path}")
    print(f"Valid predictions: {valid_count}/{len(results)}")

    if valid_count > 0:
        avg_score = np.mean([r['overall_score'] for r in results if r['valid']])
        print(f"Average toxicity score: {avg_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch prediction for molecular toxicity')
    parser.add_argument('--input', '-i', required=True, help='Input file (CSV or TXT)')
    parser.add_argument('--output', '-o', required=True, help='Output file (CSV or JSON)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()
    main(args)
