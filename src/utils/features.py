"""Feature extraction utilities for molecular graphs."""

from rdkit import Chem

# Atom features for GCN model (140 dimensions)
ATOM_FEATURES_GCN = {
    'atomic_num': list(range(1, 119)),  # 118 elements
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}

# Atom features for AttentiveFP model (148 dimensions)
ATOM_FEATURES_AFP = {
    'atomic_num': list(range(1, 119)),  # 118 elements
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
}

# Bond features (12 dimensions)
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [0, 1, 2, 3, 4, 5],
}


def one_hot(value, choices):
    """One-hot encode a value given a list of choices."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features_gcn(atom):
    """
    Get feature vector for an atom (GCN model).

    Features (140 dimensions):
    - Atomic number (one-hot, 118)
    - Degree (one-hot, 6)
    - Formal charge (one-hot, 5)
    - Hybridization (one-hot, 5)
    - Is aromatic (1)
    - Number of Hs (one-hot, 5)
    """
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_GCN['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_GCN['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_GCN['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_GCN['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_GCN['num_hs']))
    return features


def get_atom_features_afp(atom):
    """
    Get feature vector for an atom (AttentiveFP model).

    Features (148 dimensions):
    - Atomic number (one-hot, 118)
    - Degree (one-hot, 7)
    - Formal charge (one-hot, 6)
    - Chiral tag (one-hot, 4)
    - Num Hs (one-hot, 5)
    - Hybridization (one-hot, 6)
    - Is aromatic (1)
    - Is in ring (1)
    """
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_AFP['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_AFP['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_AFP['formal_charge']))
    features.extend(one_hot(int(atom.GetChiralTag()), ATOM_FEATURES_AFP['chiral_tag']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_AFP['num_hs']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_AFP['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def get_bond_features(bond):
    """
    Get feature vector for a bond/edge.

    Features (12 dimensions):
    - Bond type (one-hot, 4)
    - Stereo (one-hot, 6)
    - Is conjugated (1)
    - Is in ring (1)
    """
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo']))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features
