"""
Predict toxicity for molecules using trained model.
Shows what the model actually does in practice.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Import from shared modules
from src.models import MolecularPropertyPredictor
from src.constants import TOX21_TASKS, TASK_DESCRIPTIONS, EXAMPLE_MOLECULES, MODELS_DIR


def load_model():
    """Load trained model."""
    model = MolecularPropertyPredictor(
        input_size=2048,
        hidden_sizes=[1024, 512, 256],
        num_tasks=12,
        dropout=0.3
    )

    checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def predict_toxicity(model, smiles: str):
    """Predict toxicity for a single molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp = fp_gen.GetFingerprintAsNumPy(mol)

    with torch.no_grad():
        x = torch.FloatTensor(fp).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]

    return probs


def print_predictions(name: str, smiles: str, probs):
    """Pretty print predictions for a molecule."""
    print(f"\n{'='*60}")
    print(f"Molecule: {name}")
    print(f"SMILES: {smiles}")
    print(f"{'='*60}")

    # Sort by toxicity probability
    results = [(task, probs[i], TASK_DESCRIPTIONS[task])
               for i, task in enumerate(TOX21_TASKS)]
    results.sort(key=lambda x: x[1], reverse=True)

    # Overall toxicity score (average)
    avg_tox = np.mean(probs)

    if avg_tox > 0.5:
        risk = "HIGH RISK"
        color = "!"
    elif avg_tox > 0.3:
        risk = "MODERATE RISK"
        color = "~"
    else:
        risk = "LOW RISK"
        color = " "

    print(f"\n{color} Overall Toxicity Score: {avg_tox:.1%} ({risk})")
    print(f"\nDetailed Predictions:")
    print(f"{'-'*60}")

    for task, prob, desc in results:
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        flag = "⚠️" if prob > 0.5 else "  "
        print(f"  {flag} {task:15} [{bar}] {prob:5.1%}")
        print(f"      {desc}")


def main():
    print("Loading trained model...")
    model = load_model()
    print("Model loaded!\n")

    print("="*60)
    print("MOLECULAR TOXICITY PREDICTOR")
    print("Predicting toxicity across 12 biological targets")
    print("="*60)

    for name, smiles in EXAMPLE_MOLECULES.items():
        probs = predict_toxicity(model, smiles)
        if probs is not None:
            print_predictions(name, smiles, probs)

    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Enter a SMILES string to predict toxicity (or 'quit' to exit)")
    print("="*60)

    while True:
        try:
            smiles = input("\nEnter SMILES: ").strip()
            if smiles.lower() in ['quit', 'exit', 'q']:
                break
            if not smiles:
                continue

            probs = predict_toxicity(model, smiles)
            if probs is None:
                print("Invalid SMILES string. Please try again.")
            else:
                print_predictions("Custom Molecule", smiles, probs)
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
