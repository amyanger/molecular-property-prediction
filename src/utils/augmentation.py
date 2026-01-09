"""Data augmentation utilities for molecular data."""

import random
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem


def randomize_smiles(smiles: str, random_state: Optional[int] = None) -> Optional[str]:
    """
    Generate a random valid SMILES representation.

    The same molecule can be represented by many different SMILES strings.
    This function generates a random one for data augmentation.

    Args:
        smiles: Input SMILES string
        random_state: Random seed

    Returns:
        Randomized SMILES or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if random_state is not None:
        random.seed(random_state)

    # Randomize atom order
    atom_indices = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_indices)

    # Renumber atoms
    mol = Chem.RenumberAtoms(mol, atom_indices)

    return Chem.MolToSmiles(mol, canonical=False)


def generate_smiles_variants(
    smiles: str,
    n_variants: int = 5,
    random_state: Optional[int] = None,
) -> list[str]:
    """
    Generate multiple SMILES representations for the same molecule.

    Args:
        smiles: Input SMILES string
        n_variants: Number of variants to generate
        random_state: Random seed

    Returns:
        List of unique SMILES variants
    """
    variants = set()
    variants.add(smiles)

    # Add canonical form
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    canonical = Chem.MolToSmiles(mol, canonical=True)
    variants.add(canonical)

    # Generate random variants
    attempts = 0
    max_attempts = n_variants * 10

    while len(variants) < n_variants and attempts < max_attempts:
        seed = random_state + attempts if random_state else None
        variant = randomize_smiles(smiles, seed)
        if variant:
            variants.add(variant)
        attempts += 1

    return list(variants)[:n_variants]


def enumerate_stereoisomers(smiles: str, max_isomers: int = 10) -> list[str]:
    """
    Enumerate stereoisomers of a molecule.

    Args:
        smiles: Input SMILES string
        max_isomers: Maximum number of isomers to return

    Returns:
        List of stereoisomer SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    # Find stereocenters
    from rdkit.Chem import EnumerateStereoisomers

    opts = EnumerateStereoisomers.StereoEnumerationOptions(
        tryEmbedding=False,
        unique=True,
        maxIsomers=max_isomers,
    )

    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))

    return [Chem.MolToSmiles(iso, canonical=True) for iso in isomers]


def enumerate_tautomers(smiles: str, max_tautomers: int = 10) -> list[str]:
    """
    Enumerate tautomers of a molecule.

    Args:
        smiles: Input SMILES string
        max_tautomers: Maximum number of tautomers

    Returns:
        List of tautomer SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    from rdkit.Chem.MolStandardize import rdMolStandardize

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)

    try:
        tautomers = enumerator.Enumerate(mol)
        return [Chem.MolToSmiles(t, canonical=True) for t in tautomers]
    except Exception:
        return [smiles]


def add_noise_to_fingerprint(
    fingerprint,
    noise_level: float = 0.1,
    random_state: Optional[int] = None,
) -> list:
    """
    Add noise to a binary fingerprint for augmentation.

    Args:
        fingerprint: Binary fingerprint (list or array)
        noise_level: Fraction of bits to flip
        random_state: Random seed

    Returns:
        Noisy fingerprint
    """
    import numpy as np

    if random_state is not None:
        np.random.seed(random_state)

    fp = np.array(fingerprint).copy()
    n_bits = len(fp)
    n_flip = int(n_bits * noise_level)

    flip_indices = np.random.choice(n_bits, n_flip, replace=False)
    fp[flip_indices] = 1 - fp[flip_indices]

    return fp.tolist()


def augment_graph_dropout(
    x,
    edge_index,
    node_dropout: float = 0.1,
    edge_dropout: float = 0.1,
    random_state: Optional[int] = None,
):
    """
    Apply dropout augmentation to molecular graph.

    Args:
        x: Node features tensor
        edge_index: Edge indices tensor
        node_dropout: Fraction of node features to zero
        edge_dropout: Fraction of edges to remove
        random_state: Random seed

    Returns:
        Augmented (x, edge_index)
    """
    import torch

    if random_state is not None:
        torch.manual_seed(random_state)

    # Node feature dropout
    if node_dropout > 0:
        mask = torch.rand(x.shape) > node_dropout
        x = x * mask.float()

    # Edge dropout
    if edge_dropout > 0:
        n_edges = edge_index.shape[1]
        keep_mask = torch.rand(n_edges) > edge_dropout
        edge_index = edge_index[:, keep_mask]

    return x, edge_index


class MolecularAugmenter:
    """
    Molecular data augmentation pipeline.

    Usage:
        augmenter = MolecularAugmenter(
            smiles_variants=3,
            include_stereoisomers=True,
            fingerprint_noise=0.05
        )
        augmented_smiles = augmenter.augment_smiles("CCO")
        augmented_fp = augmenter.augment_fingerprint(fp)
    """

    def __init__(
        self,
        smiles_variants: int = 3,
        include_stereoisomers: bool = False,
        include_tautomers: bool = False,
        fingerprint_noise: float = 0.0,
        graph_node_dropout: float = 0.0,
        graph_edge_dropout: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.smiles_variants = smiles_variants
        self.include_stereoisomers = include_stereoisomers
        self.include_tautomers = include_tautomers
        self.fingerprint_noise = fingerprint_noise
        self.graph_node_dropout = graph_node_dropout
        self.graph_edge_dropout = graph_edge_dropout
        self.random_state = random_state

    def augment_smiles(self, smiles: str) -> list[str]:
        """Generate augmented SMILES representations."""
        augmented = set()

        # Add original and variants
        augmented.update(generate_smiles_variants(
            smiles, self.smiles_variants, self.random_state
        ))

        # Add stereoisomers
        if self.include_stereoisomers:
            augmented.update(enumerate_stereoisomers(smiles, max_isomers=5))

        # Add tautomers
        if self.include_tautomers:
            augmented.update(enumerate_tautomers(smiles, max_tautomers=5))

        return list(augmented)

    def augment_fingerprint(self, fingerprint) -> list:
        """Add noise to fingerprint."""
        if self.fingerprint_noise > 0:
            return add_noise_to_fingerprint(
                fingerprint, self.fingerprint_noise, self.random_state
            )
        return fingerprint

    def augment_graph(self, x, edge_index):
        """Apply graph augmentation."""
        return augment_graph_dropout(
            x, edge_index,
            self.graph_node_dropout,
            self.graph_edge_dropout,
            self.random_state
        )


def create_augmented_dataset(
    smiles_list: list[str],
    labels,
    augmenter: MolecularAugmenter,
) -> tuple[list[str], list]:
    """
    Create augmented dataset from original data.

    Args:
        smiles_list: List of SMILES strings
        labels: Labels for each molecule
        augmenter: MolecularAugmenter instance

    Returns:
        Tuple of (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smi, label in zip(smiles_list, labels):
        variants = augmenter.augment_smiles(smi)

        for variant in variants:
            augmented_smiles.append(variant)
            augmented_labels.append(label)

    return augmented_smiles, augmented_labels
