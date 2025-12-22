"""
Predict toxicity for molecules using trained model.
Shows what the model actually does in practice.
"""

import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]

# Human-readable descriptions
TASK_DESCRIPTIONS = {
    "NR-AR": "Androgen Receptor (hormone disruption)",
    "NR-AR-LBD": "Androgen Receptor Binding",
    "NR-AhR": "Aryl Hydrocarbon Receptor (dioxin-like toxicity)",
    "NR-Aromatase": "Aromatase Enzyme (estrogen synthesis)",
    "NR-ER": "Estrogen Receptor (hormone disruption)",
    "NR-ER-LBD": "Estrogen Receptor Binding",
    "NR-PPAR-gamma": "PPAR-gamma (metabolic effects)",
    "SR-ARE": "Antioxidant Response (oxidative stress)",
    "SR-ATAD5": "DNA Damage Response",
    "SR-HSE": "Heat Shock Response (cellular stress)",
    "SR-MMP": "Mitochondrial Toxicity",
    "SR-p53": "Tumor Suppressor p53 (cancer risk)"
}

# Example molecules to test
EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol (Alcohol)": "CCO",
    "Nicotine": "CN1CCCC1C2=CN=CC=C2",
    "Acetaminophen (Tylenol)": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Benzene (toxic)": "C1=CC=CC=C1",
    "Formaldehyde (toxic)": "C=O",
    "DDT (pesticide)": "ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2",
    "Vitamin C": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
}


def load_model():
    """Load trained model."""
    from train import MolecularPropertyPredictor

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
