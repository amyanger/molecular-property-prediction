"""
Download and prepare molecular property prediction datasets.
Downloads directly from MoleculeNet without DeepChem's TensorFlow dependency.
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS = {
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "description": "Toxicity prediction across 12 biological targets",
        "task_type": "classification",
        "tasks": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
            "SR-HSE", "SR-MMP", "SR-p53"
        ]
    },
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "description": "Blood-brain barrier penetration prediction",
        "task_type": "classification",
        "tasks": ["p_np"]
    },
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "description": "Aqueous solubility prediction",
        "task_type": "regression",
        "tasks": ["measured log solubility in mols per litre"]
    },
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "description": "Lipophilicity (octanol/water partition coefficient)",
        "task_type": "regression",
        "tasks": ["exp"]
    }
}


def download_dataset(name: str):
    """Download a single dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[name]
    url = info["url"]

    # Create directory
    dataset_dir = DATA_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    filename = url.split("/")[-1]
    filepath = dataset_dir / filename

    print(f"\nDownloading {name}...")
    print(f"  URL: {url}")
    print(f"  Description: {info['description']}")

    if filepath.exists():
        print(f"  Already exists: {filepath}")
    else:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to: {filepath}")

    # Load and display stats
    if filename.endswith('.gz'):
        df = pd.read_csv(filepath, compression='gzip')
    else:
        df = pd.read_csv(filepath)

    # Find SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'mol']:
        if col in df.columns:
            smiles_col = col
            break

    print(f"\n  Dataset Statistics:")
    print(f"    Total samples: {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    SMILES column: {smiles_col}")
    print(f"    Task type: {info['task_type']}")
    print(f"    Tasks: {info['tasks']}")

    # Show sample
    if smiles_col:
        print(f"\n  Sample molecules:")
        for i, row in df.head(3).iterrows():
            print(f"    {row[smiles_col][:50]}...")

    return df


def download_all():
    """Download all datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Molecular Property Prediction - Dataset Download")
    print("=" * 60)

    datasets = {}
    for name in DATASETS:
        datasets[name] = download_dataset(name)
        print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("All datasets downloaded successfully!")
    print(f"Data stored in: {DATA_DIR.absolute()}")

    return datasets


if __name__ == "__main__":
    download_all()
