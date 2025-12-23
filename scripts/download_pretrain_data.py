"""
Download large molecular datasets for pre-training GNN models.

Datasets:
- ZINC: 250K drug-like molecules (default)
- ChEMBL: Bioactive molecules (optional, requires chembl_webresource_client)

Usage:
    python scripts/download_pretrain_data.py
    python scripts/download_pretrain_data.py --dataset zinc --subset  # Smaller subset
    python scripts/download_pretrain_data.py --dataset chembl --num_molecules 50000
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from src.constants import DATA_DIR


def download_zinc(subset=False):
    """
    Download ZINC dataset using PyTorch Geometric.

    Args:
        subset: If True, download smaller subset (12K), else full (250K)

    Returns:
        Path to saved SMILES file
    """
    from torch_geometric.datasets import ZINC

    print("Downloading ZINC dataset...")
    zinc_dir = DATA_DIR / "zinc"
    zinc_dir.mkdir(parents=True, exist_ok=True)

    # Download via PyG (handles caching automatically)
    dataset = ZINC(root=str(zinc_dir), subset=subset, split='train')

    size_name = "subset (12K)" if subset else "full (250K)"
    print(f"Downloaded ZINC {size_name}: {len(dataset)} molecules")

    # ZINC in PyG doesn't have SMILES directly, so we note this
    print(f"Dataset saved to: {zinc_dir}")
    print("Note: ZINC dataset is stored in PyG format (graph tensors)")

    return zinc_dir


def download_zinc_smiles(num_molecules=50000):
    """
    Download ZINC SMILES from the ZINC database directly.
    This gives us actual SMILES strings we can convert to our feature format.
    """
    import urllib.request

    zinc_dir = DATA_DIR / "zinc"
    zinc_dir.mkdir(parents=True, exist_ok=True)

    # ZINC tranches - small drug-like molecules
    # Using ZINC15 subset URLs
    urls = [
        "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
    ]

    smiles_list = []

    for url in urls:
        print(f"Downloading from {url}...")
        try:
            df = pd.read_csv(url)
            if 'smiles' in df.columns:
                smiles_list.extend(df['smiles'].tolist())
            elif 'SMILES' in df.columns:
                smiles_list.extend(df['SMILES'].tolist())
            else:
                # First column is usually SMILES
                smiles_list.extend(df.iloc[:, 0].tolist())
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    if len(smiles_list) == 0:
        print("Could not download SMILES. Using PyG ZINC dataset instead.")
        return download_zinc(subset=True)

    # Validate SMILES
    print("Validating SMILES...")
    valid_smiles = []
    for smi in tqdm(smiles_list[:num_molecules]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumAtoms() > 0:
            valid_smiles.append(smi)

    print(f"Valid SMILES: {len(valid_smiles)}")

    # Save to CSV
    output_path = zinc_dir / "zinc_pretrain.csv"
    pd.DataFrame({'smiles': valid_smiles}).to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return output_path


def download_chembl(num_molecules=50000):
    """
    Download molecules from ChEMBL database.
    Requires: pip install chembl_webresource_client
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("ChEMBL client not installed. Install with:")
        print("  pip install chembl_webresource_client")
        print("Falling back to ZINC dataset...")
        return download_zinc_smiles(num_molecules)

    chembl_dir = DATA_DIR / "chembl"
    chembl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {num_molecules} molecules from ChEMBL...")

    molecule = new_client.molecule

    # Filter for drug-like molecules (Lipinski's Rule of 5)
    mols = molecule.filter(
        molecule_properties__mw_freebase__lte=500,
        molecule_properties__mw_freebase__gte=100,
        molecule_properties__alogp__lte=5,
        molecule_properties__alogp__gte=-2,
    )

    smiles_list = []
    print("Downloading molecules...")

    for mol in tqdm(mols, total=num_molecules):
        if len(smiles_list) >= num_molecules:
            break

        if mol.get('molecule_structures'):
            smi = mol['molecule_structures'].get('canonical_smiles')
            if smi:
                # Validate
                rdkit_mol = Chem.MolFromSmiles(smi)
                if rdkit_mol is not None and rdkit_mol.GetNumAtoms() > 0:
                    smiles_list.append(smi)

    print(f"Downloaded {len(smiles_list)} valid molecules")

    # Save to CSV
    output_path = chembl_dir / "chembl_pretrain.csv"
    pd.DataFrame({'smiles': smiles_list}).to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return output_path


def download_pubchem(num_molecules=50000):
    """
    Download molecules from PubChem (alternative large dataset).
    Uses pre-compiled drug-like subset.
    """
    import urllib.request

    pubchem_dir = DATA_DIR / "pubchem"
    pubchem_dir.mkdir(parents=True, exist_ok=True)

    # PubChem provides compound downloads
    # For simplicity, we use a curated subset
    print("PubChem download not yet implemented. Using ZINC instead.")
    return download_zinc_smiles(num_molecules)


def main(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'zinc':
        if args.smiles:
            path = download_zinc_smiles(args.num_molecules)
        else:
            path = download_zinc(subset=args.subset)
    elif args.dataset == 'chembl':
        path = download_chembl(args.num_molecules)
    elif args.dataset == 'pubchem':
        path = download_pubchem(args.num_molecules)
    else:
        print(f"Unknown dataset: {args.dataset}")
        return

    print(f"\nPre-training data ready at: {path}")
    print("\nNext step: Run pre-training with:")
    print("  python scripts/pretrain_gnn.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download pre-training datasets')
    parser.add_argument('--dataset', type=str, default='zinc',
                        choices=['zinc', 'chembl', 'pubchem'],
                        help='Dataset to download')
    parser.add_argument('--subset', action='store_true',
                        help='Use smaller subset (ZINC only)')
    parser.add_argument('--smiles', action='store_true', default=True,
                        help='Download as SMILES CSV (recommended)')
    parser.add_argument('--num_molecules', type=int, default=50000,
                        help='Number of molecules to download')

    args = parser.parse_args()
    main(args)
