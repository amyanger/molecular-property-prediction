"""Shared constants for molecular property prediction."""

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

# Tox21 task names (12 toxicity endpoints)
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]

# Human-readable descriptions for each task
TASK_DESCRIPTIONS = {
    "NR-AR": "Androgen Receptor - hormone disruption",
    "NR-AR-LBD": "Androgen Receptor Binding Domain",
    "NR-AhR": "Aryl Hydrocarbon Receptor - dioxin-like toxicity",
    "NR-Aromatase": "Aromatase Enzyme - estrogen synthesis disruption",
    "NR-ER": "Estrogen Receptor - hormone disruption",
    "NR-ER-LBD": "Estrogen Receptor Binding Domain",
    "NR-PPAR-gamma": "PPAR-gamma - metabolic effects",
    "SR-ARE": "Antioxidant Response Element - oxidative stress",
    "SR-ATAD5": "ATAD5 - DNA damage response",
    "SR-HSE": "Heat Shock Response - cellular stress",
    "SR-MMP": "Mitochondrial Membrane Potential - mitochondrial toxicity",
    "SR-p53": "p53 Tumor Suppressor - cancer risk"
}

# Example molecules for testing
EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Acetaminophen (Tylenol)": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Nicotine": "CN1CCCC1C2=CN=CC=C2",
    "Ethanol": "CCO",
    "Benzene (toxic)": "C1=CC=CC=C1",
    "DDT (pesticide)": "ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2",
    "Formaldehyde (toxic)": "C=O",
    "Vitamin C": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
}

# Default ensemble weights (optimized)
DEFAULT_ENSEMBLE_WEIGHTS = {
    'mlp': 0.1,
    'gcn': 0.5,
    'attentivefp': 0.4
}
