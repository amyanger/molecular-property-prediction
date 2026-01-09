"""Molecular descriptor calculation utilities."""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D, rdMolDescriptors, Lipinski
from rdkit.Chem import AllChem
import logging

logger = logging.getLogger(__name__)


@dataclass
class MolecularDescriptors:
    """Container for molecular descriptors."""

    smiles: str

    # Constitutional descriptors
    molecular_weight: float = 0.0
    heavy_atom_count: int = 0
    num_atoms: int = 0
    num_bonds: int = 0
    num_rotatable_bonds: int = 0

    # Topological descriptors
    num_rings: int = 0
    num_aromatic_rings: int = 0
    num_heteroatoms: int = 0
    num_hbd: int = 0  # H-bond donors
    num_hba: int = 0  # H-bond acceptors

    # Electronic descriptors
    logp: float = 0.0
    tpsa: float = 0.0  # Topological polar surface area
    mr: float = 0.0  # Molar refractivity

    # Atom counts
    num_carbons: int = 0
    num_nitrogens: int = 0
    num_oxygens: int = 0
    num_sulfurs: int = 0
    num_halogens: int = 0

    # Complexity
    bertz_ct: float = 0.0  # Bertz complexity
    fraction_csp3: float = 0.0  # Fraction of sp3 carbons

    # Ring descriptors
    num_aliphatic_rings: int = 0
    num_heterocycles: int = 0
    num_saturated_rings: int = 0

    # Surface area descriptors
    labute_asa: float = 0.0  # Labute's approximate surface area

    # All descriptors as numpy array
    descriptor_vector: np.ndarray = field(default_factory=lambda: np.array([]))


class DescriptorCalculator:
    """
    Calculate molecular descriptors from SMILES.

    Computes a comprehensive set of physicochemical and topological
    descriptors for use in machine learning models.
    """

    # RDKit descriptor functions
    RDKIT_DESCRIPTORS = {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
        "ExactMolWt": Descriptors.ExactMolWt,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinPartialCharge": Descriptors.MinPartialCharge,
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
        "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
        "FpDensityMorgan3": Descriptors.FpDensityMorgan3,
        "BCUT2D_MWHI": Descriptors.BCUT2D_MWHI,
        "BCUT2D_MWLOW": Descriptors.BCUT2D_MWLOW,
        "BCUT2D_CHGHI": Descriptors.BCUT2D_CHGHI,
        "BCUT2D_CHGLO": Descriptors.BCUT2D_CHGLO,
        "BCUT2D_LOGPHI": Descriptors.BCUT2D_LOGPHI,
        "BCUT2D_LOGPLOW": Descriptors.BCUT2D_LOGPLOW,
        "BCUT2D_MRHI": Descriptors.BCUT2D_MRHI,
        "BCUT2D_MRLOW": Descriptors.BCUT2D_MRLOW,
        "BalabanJ": Descriptors.BalabanJ,
        "BertzCT": Descriptors.BertzCT,
        "Chi0": Descriptors.Chi0,
        "Chi0n": Descriptors.Chi0n,
        "Chi0v": Descriptors.Chi0v,
        "Chi1": Descriptors.Chi1,
        "Chi1n": Descriptors.Chi1n,
        "Chi1v": Descriptors.Chi1v,
        "Chi2n": Descriptors.Chi2n,
        "Chi2v": Descriptors.Chi2v,
        "Chi3n": Descriptors.Chi3n,
        "Chi3v": Descriptors.Chi3v,
        "Chi4n": Descriptors.Chi4n,
        "Chi4v": Descriptors.Chi4v,
        "HallKierAlpha": Descriptors.HallKierAlpha,
        "Ipc": Descriptors.Ipc,
        "Kappa1": Descriptors.Kappa1,
        "Kappa2": Descriptors.Kappa2,
        "Kappa3": Descriptors.Kappa3,
        "LabuteASA": Descriptors.LabuteASA,
        "PEOE_VSA1": Descriptors.PEOE_VSA1,
        "PEOE_VSA10": Descriptors.PEOE_VSA10,
        "PEOE_VSA11": Descriptors.PEOE_VSA11,
        "PEOE_VSA12": Descriptors.PEOE_VSA12,
        "PEOE_VSA13": Descriptors.PEOE_VSA13,
        "PEOE_VSA14": Descriptors.PEOE_VSA14,
        "PEOE_VSA2": Descriptors.PEOE_VSA2,
        "PEOE_VSA3": Descriptors.PEOE_VSA3,
        "PEOE_VSA4": Descriptors.PEOE_VSA4,
        "PEOE_VSA5": Descriptors.PEOE_VSA5,
        "PEOE_VSA6": Descriptors.PEOE_VSA6,
        "PEOE_VSA7": Descriptors.PEOE_VSA7,
        "PEOE_VSA8": Descriptors.PEOE_VSA8,
        "PEOE_VSA9": Descriptors.PEOE_VSA9,
        "SMR_VSA1": Descriptors.SMR_VSA1,
        "SMR_VSA10": Descriptors.SMR_VSA10,
        "SMR_VSA2": Descriptors.SMR_VSA2,
        "SMR_VSA3": Descriptors.SMR_VSA3,
        "SMR_VSA4": Descriptors.SMR_VSA4,
        "SMR_VSA5": Descriptors.SMR_VSA5,
        "SMR_VSA6": Descriptors.SMR_VSA6,
        "SMR_VSA7": Descriptors.SMR_VSA7,
        "SMR_VSA8": Descriptors.SMR_VSA8,
        "SMR_VSA9": Descriptors.SMR_VSA9,
        "SlogP_VSA1": Descriptors.SlogP_VSA1,
        "SlogP_VSA10": Descriptors.SlogP_VSA10,
        "SlogP_VSA11": Descriptors.SlogP_VSA11,
        "SlogP_VSA12": Descriptors.SlogP_VSA12,
        "SlogP_VSA2": Descriptors.SlogP_VSA2,
        "SlogP_VSA3": Descriptors.SlogP_VSA3,
        "SlogP_VSA4": Descriptors.SlogP_VSA4,
        "SlogP_VSA5": Descriptors.SlogP_VSA5,
        "SlogP_VSA6": Descriptors.SlogP_VSA6,
        "SlogP_VSA7": Descriptors.SlogP_VSA7,
        "SlogP_VSA8": Descriptors.SlogP_VSA8,
        "SlogP_VSA9": Descriptors.SlogP_VSA9,
        "TPSA": Descriptors.TPSA,
        "EState_VSA1": Descriptors.EState_VSA1,
        "EState_VSA10": Descriptors.EState_VSA10,
        "EState_VSA11": Descriptors.EState_VSA11,
        "EState_VSA2": Descriptors.EState_VSA2,
        "EState_VSA3": Descriptors.EState_VSA3,
        "EState_VSA4": Descriptors.EState_VSA4,
        "EState_VSA5": Descriptors.EState_VSA5,
        "EState_VSA6": Descriptors.EState_VSA6,
        "EState_VSA7": Descriptors.EState_VSA7,
        "EState_VSA8": Descriptors.EState_VSA8,
        "EState_VSA9": Descriptors.EState_VSA9,
        "VSA_EState1": Descriptors.VSA_EState1,
        "VSA_EState10": Descriptors.VSA_EState10,
        "VSA_EState2": Descriptors.VSA_EState2,
        "VSA_EState3": Descriptors.VSA_EState3,
        "VSA_EState4": Descriptors.VSA_EState4,
        "VSA_EState5": Descriptors.VSA_EState5,
        "VSA_EState6": Descriptors.VSA_EState6,
        "VSA_EState7": Descriptors.VSA_EState7,
        "VSA_EState8": Descriptors.VSA_EState8,
        "VSA_EState9": Descriptors.VSA_EState9,
        "FractionCSP3": Descriptors.FractionCSP3,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "NHOHCount": Descriptors.NHOHCount,
        "NOCount": Descriptors.NOCount,
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "RingCount": Descriptors.RingCount,
        "MolLogP": Descriptors.MolLogP,
        "MolMR": Descriptors.MolMR,
        "qed": Descriptors.qed,
    }

    def __init__(self, descriptor_subset: Optional[list[str]] = None):
        """
        Initialize descriptor calculator.

        Args:
            descriptor_subset: Optional list of descriptor names to compute
        """
        if descriptor_subset:
            self.descriptors = {
                name: func for name, func in self.RDKIT_DESCRIPTORS.items()
                if name in descriptor_subset
            }
        else:
            self.descriptors = self.RDKIT_DESCRIPTORS

        self.descriptor_names = list(self.descriptors.keys())

    def calculate(self, smiles: str) -> Optional[MolecularDescriptors]:
        """
        Calculate descriptors for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            MolecularDescriptors or None if invalid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        result = MolecularDescriptors(smiles=smiles)

        # Constitutional
        result.molecular_weight = Descriptors.MolWt(mol)
        result.heavy_atom_count = mol.GetNumHeavyAtoms()
        result.num_atoms = mol.GetNumAtoms()
        result.num_bonds = mol.GetNumBonds()
        result.num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Topological
        result.num_rings = rdMolDescriptors.CalcNumRings(mol)
        result.num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        result.num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        result.num_hbd = rdMolDescriptors.CalcNumHBD(mol)
        result.num_hba = rdMolDescriptors.CalcNumHBA(mol)

        # Electronic
        result.logp = Descriptors.MolLogP(mol)
        result.tpsa = Descriptors.TPSA(mol)
        result.mr = Descriptors.MolMR(mol)

        # Atom counts
        result.num_carbons = self._count_atom(mol, 6)
        result.num_nitrogens = self._count_atom(mol, 7)
        result.num_oxygens = self._count_atom(mol, 8)
        result.num_sulfurs = self._count_atom(mol, 16)
        result.num_halogens = sum(
            self._count_atom(mol, n) for n in [9, 17, 35, 53]
        )

        # Complexity
        result.bertz_ct = Descriptors.BertzCT(mol)
        result.fraction_csp3 = Descriptors.FractionCSP3(mol)

        # Ring descriptors
        result.num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
        result.num_heterocycles = Descriptors.NumAliphaticHeterocycles(mol) + \
                                  Descriptors.NumAromaticHeterocycles(mol)
        result.num_saturated_rings = Descriptors.NumSaturatedRings(mol)

        # Surface area
        result.labute_asa = Descriptors.LabuteASA(mol)

        # Compute full descriptor vector
        result.descriptor_vector = self.calculate_vector(smiles)

        return result

    def calculate_vector(self, smiles: str) -> np.ndarray:
        """
        Calculate descriptor vector for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Numpy array of descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(self.descriptors))

        values = []
        for name, func in self.descriptors.items():
            try:
                value = func(mol)
                if value is None or np.isnan(value) or np.isinf(value):
                    value = 0.0
            except Exception:
                value = 0.0
            values.append(value)

        return np.array(values, dtype=np.float32)

    def calculate_batch(self, smiles_list: list[str]) -> np.ndarray:
        """
        Calculate descriptors for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Numpy array of shape (n_molecules, n_descriptors)
        """
        vectors = []
        for smiles in smiles_list:
            vectors.append(self.calculate_vector(smiles))
        return np.stack(vectors)

    def _count_atom(self, mol: Chem.Mol, atomic_num: int) -> int:
        """Count atoms of a specific type."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)

    def get_descriptor_names(self) -> list[str]:
        """Get list of descriptor names."""
        return self.descriptor_names


def calculate_mordred_descriptors(smiles: str) -> Optional[np.ndarray]:
    """
    Calculate Mordred descriptors (if available).

    Args:
        smiles: SMILES string

    Returns:
        Descriptor array or None
    """
    try:
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        result = calc(mol)
        return np.array([float(v) if v is not None else 0.0 for v in result])
    except ImportError:
        logger.warning("Mordred not installed. Install with: pip install mordred")
        return None


def get_descriptor_subset(subset_name: str) -> list[str]:
    """
    Get predefined subsets of descriptors.

    Args:
        subset_name: Name of subset ('physicochemical', 'topological', 'all')

    Returns:
        List of descriptor names
    """
    subsets = {
        "physicochemical": [
            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "FractionCSP3", "MolMR", "qed",
        ],
        "topological": [
            "BalabanJ", "BertzCT", "Chi0", "Chi1", "Chi2n", "Chi3n",
            "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", "RingCount",
        ],
        "counts": [
            "HeavyAtomCount", "NumHeteroatoms", "NumAromaticRings",
            "NumAliphaticRings", "NumSaturatedRings", "NHOHCount", "NOCount",
        ],
        "all": list(DescriptorCalculator.RDKIT_DESCRIPTORS.keys()),
    }

    if subset_name not in subsets:
        raise ValueError(f"Unknown subset: {subset_name}. Available: {list(subsets.keys())}")

    return subsets[subset_name]
