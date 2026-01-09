#!/usr/bin/env python
"""
Command-line interface for molecular property prediction.

Usage:
    molprop predict "CCO"
    molprop predict "CCO" --model mlp
    molprop validate "CC(=O)OC1=CC=CC=C1C(=O)O"
    molprop info "c1ccccc1"
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.models import MolecularPropertyPredictor, GNN, AttentiveFPPredictor
from src.constants import TOX21_TASKS, TASK_DESCRIPTIONS, MODELS_DIR, DEFAULT_ENSEMBLE_WEIGHTS
from src.utils.features import get_atom_features_gcn, get_atom_features_afp, get_bond_features
from src.utils.smiles import validate_smiles, get_molecule_info, calculate_drug_likeness


def get_risk_level(prob):
    """Get risk level string from probability."""
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MODERATE"
    else:
        return "HIGH"


def get_risk_color(level):
    """Get ANSI color code for risk level."""
    colors = {
        "LOW": "\033[92m",      # Green
        "MODERATE": "\033[93m", # Yellow
        "HIGH": "\033[91m",     # Red
    }
    return colors.get(level, "")


RESET = "\033[0m"
BOLD = "\033[1m"


def load_models(device):
    """Load all trained models."""
    models = {}

    try:
        mlp = MolecularPropertyPredictor(input_size=2048, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', map_location=device, weights_only=False)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device).eval()
        models['mlp'] = mlp
    except Exception:
        pass

    try:
        gnn = GNN(num_node_features=141, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', map_location=device, weights_only=False)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        gnn.to(device).eval()
        models['gnn'] = gnn
    except Exception:
        pass

    try:
        afp = AttentiveFPPredictor(in_channels=148, out_channels=12, edge_dim=12)
        checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', map_location=device, weights_only=False)
        afp.load_state_dict(checkpoint['model_state_dict'])
        afp.to(device).eval()
        models['attentivefp'] = afp
    except Exception:
        pass

    return models


def predict_ensemble(smiles, models, device):
    """Get ensemble prediction."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    predictions = []
    weights = []

    # MLP
    if 'mlp' in models:
        fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp = fp_gen.GetFingerprintAsNumPy(mol)
        x = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(models['mlp'](x)).cpu().numpy()[0]
        predictions.append(probs)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['mlp'])

    # GNN
    if 'gnn' in models:
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

        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        with torch.no_grad():
            probs = torch.sigmoid(models['gnn'](x, edge_index, batch)).cpu().numpy()[0]
        predictions.append(probs)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['gcn'])

    # AttentiveFP
    if 'attentivefp' in models:
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

        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)
        with torch.no_grad():
            probs = torch.sigmoid(models['attentivefp'](x, edge_index, edge_attr, batch)).cpu().numpy()[0]
        predictions.append(probs)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['attentivefp'])

    if not predictions:
        return None

    weights = np.array(weights) / sum(weights)
    return sum(w * p for w, p in zip(weights, predictions))


def cmd_predict(args):
    """Handle predict command."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Validate SMILES
    is_valid, error = validate_smiles(args.smiles)
    if not is_valid:
        print(f"{BOLD}Error:{RESET} {error}")
        sys.exit(1)

    # Load models
    models = load_models(device)
    if not models:
        print(f"{BOLD}Error:{RESET} No trained models found. Run training first.")
        sys.exit(1)

    # Get prediction
    probs = predict_ensemble(args.smiles, models, device)
    if probs is None:
        print(f"{BOLD}Error:{RESET} Prediction failed")
        sys.exit(1)

    # Output
    if args.json:
        result = {
            'smiles': args.smiles,
            'overall_score': float(np.mean(probs)),
            'predictions': {task: float(probs[i]) for i, task in enumerate(TOX21_TASKS)}
        }
        print(json.dumps(result, indent=2))
    else:
        overall = np.mean(probs)
        risk = get_risk_level(overall)
        color = get_risk_color(risk)

        print(f"\n{BOLD}Toxicity Prediction for:{RESET} {args.smiles}")
        print("=" * 60)
        print(f"{BOLD}Overall Score:{RESET} {color}{overall:.4f} ({risk}){RESET}")
        print()
        print(f"{BOLD}Per-Endpoint Predictions:{RESET}")

        for i, task in enumerate(TOX21_TASKS):
            prob = probs[i]
            risk_level = get_risk_level(prob)
            risk_color = get_risk_color(risk_level)
            desc = TASK_DESCRIPTIONS.get(task, "")
            print(f"  {task:15} {risk_color}{prob:.4f}{RESET} ({risk_level:8}) - {desc}")


def cmd_validate(args):
    """Handle validate command."""
    is_valid, error = validate_smiles(args.smiles)

    if args.json:
        print(json.dumps({'smiles': args.smiles, 'valid': is_valid, 'error': error}))
    else:
        if is_valid:
            print(f"{BOLD}Valid:{RESET} {args.smiles}")
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles), canonical=True)
            print(f"{BOLD}Canonical:{RESET} {canonical}")
        else:
            print(f"{BOLD}Invalid:{RESET} {error}")


def cmd_info(args):
    """Handle info command."""
    info = get_molecule_info(args.smiles)

    if args.json:
        result = {
            'smiles': info.smiles,
            'canonical_smiles': info.canonical_smiles,
            'valid': info.is_valid,
            'molecular_weight': info.molecular_weight,
            'num_atoms': info.num_atoms,
            'num_heavy_atoms': info.num_heavy_atoms,
            'num_bonds': info.num_bonds,
            'num_rings': info.num_rings,
            'num_aromatic_rings': info.num_aromatic_rings,
            'logp': info.logp,
            'tpsa': info.tpsa,
            'hbd': info.hbd,
            'hba': info.hba,
        }
        print(json.dumps(result, indent=2))
    else:
        if not info.is_valid:
            print(f"{BOLD}Error:{RESET} {info.error_message}")
            sys.exit(1)

        print(f"\n{BOLD}Molecule Information{RESET}")
        print("=" * 40)
        print(f"Input SMILES:     {info.smiles}")
        print(f"Canonical SMILES: {info.canonical_smiles}")
        print()
        print(f"{BOLD}Structure:{RESET}")
        print(f"  Atoms:          {info.num_atoms}")
        print(f"  Heavy atoms:    {info.num_heavy_atoms}")
        print(f"  Bonds:          {info.num_bonds}")
        print(f"  Rings:          {info.num_rings}")
        print(f"  Aromatic rings: {info.num_aromatic_rings}")
        print()
        print(f"{BOLD}Properties:{RESET}")
        print(f"  Molecular weight: {info.molecular_weight:.2f}")
        print(f"  LogP:             {info.logp:.2f}")
        print(f"  TPSA:             {info.tpsa:.2f}")
        print(f"  HBD:              {info.hbd}")
        print(f"  HBA:              {info.hba}")

        # Drug-likeness
        drug_like = calculate_drug_likeness(args.smiles)
        print()
        print(f"{BOLD}Drug-likeness (Lipinski):{RESET}")
        print(f"  Violations: {drug_like['lipinski_violations']}")
        if drug_like['violation_details']:
            for v in drug_like['violation_details']:
                print(f"    - {v}")
        print(f"  Drug-like: {'Yes' if drug_like['is_drug_like'] else 'No'}")


def cmd_tasks(args):
    """Handle tasks command."""
    if args.json:
        result = [{'name': t, 'description': TASK_DESCRIPTIONS.get(t, '')} for t in TOX21_TASKS]
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{BOLD}Tox21 Toxicity Endpoints{RESET}")
        print("=" * 60)
        for task in TOX21_TASKS:
            print(f"  {task:15} - {TASK_DESCRIPTIONS.get(task, '')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='molprop',
        description='Molecular Property Prediction CLI',
    )
    parser.add_argument('--version', action='version', version='molprop 1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict toxicity')
    predict_parser.add_argument('smiles', help='SMILES string')
    predict_parser.add_argument('--model', default='ensemble', choices=['mlp', 'gnn', 'attentivefp', 'ensemble'])
    predict_parser.add_argument('--json', action='store_true', help='JSON output')
    predict_parser.add_argument('--cpu', action='store_true', help='Force CPU')
    predict_parser.set_defaults(func=cmd_predict)

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate SMILES')
    validate_parser.add_argument('smiles', help='SMILES string')
    validate_parser.add_argument('--json', action='store_true', help='JSON output')
    validate_parser.set_defaults(func=cmd_validate)

    # Info command
    info_parser = subparsers.add_parser('info', help='Get molecule info')
    info_parser.add_argument('smiles', help='SMILES string')
    info_parser.add_argument('--json', action='store_true', help='JSON output')
    info_parser.set_defaults(func=cmd_info)

    # Tasks command
    tasks_parser = subparsers.add_parser('tasks', help='List toxicity tasks')
    tasks_parser.add_argument('--json', action='store_true', help='JSON output')
    tasks_parser.set_defaults(func=cmd_tasks)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
