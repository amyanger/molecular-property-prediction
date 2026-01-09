"""SMILES string validation and processing utilities."""

from dataclasses import dataclass
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


@dataclass
class MoleculeInfo:
    """Information about a validated molecule."""

    smiles: str
    canonical_smiles: str
    is_valid: bool
    num_atoms: int
    num_heavy_atoms: int
    num_bonds: int
    molecular_weight: float
    num_rings: int
    num_aromatic_rings: int
    num_rotatable_bonds: int
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    logp: float  # Predicted logP
    error_message: Optional[str] = None


def validate_smiles(smiles: str) -> tuple[bool, Optional[str]]:
    """
    Validate a SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES string is empty or not a string"

    smiles = smiles.strip()

    if len(smiles) == 0:
        return False, "SMILES string is empty"

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False, "Invalid SMILES: RDKit could not parse the molecule"

    # Check for disconnected fragments (salts, mixtures)
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        return False, f"SMILES contains {len(frags)} disconnected fragments"

    return True, None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form.

    Args:
        smiles: SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_molecule_info(smiles: str) -> MoleculeInfo:
    """
    Get detailed information about a molecule from its SMILES.

    Args:
        smiles: SMILES string

    Returns:
        MoleculeInfo dataclass with molecule properties
    """
    is_valid, error_msg = validate_smiles(smiles)

    if not is_valid:
        return MoleculeInfo(
            smiles=smiles,
            canonical_smiles="",
            is_valid=False,
            num_atoms=0,
            num_heavy_atoms=0,
            num_bonds=0,
            molecular_weight=0.0,
            num_rings=0,
            num_aromatic_rings=0,
            num_rotatable_bonds=0,
            hbd=0,
            hba=0,
            tpsa=0.0,
            logp=0.0,
            error_message=error_msg,
        )

    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)

    return MoleculeInfo(
        smiles=smiles,
        canonical_smiles=canonical,
        is_valid=True,
        num_atoms=mol.GetNumAtoms(),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_bonds=mol.GetNumBonds(),
        molecular_weight=round(Descriptors.MolWt(mol), 2),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        hbd=rdMolDescriptors.CalcNumHBD(mol),
        hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 2),
        error_message=None,
    )


def standardize_smiles(smiles: str) -> Optional[str]:
    """
    Standardize a SMILES string (remove salts, neutralize, canonicalize).

    Args:
        smiles: SMILES string

    Returns:
        Standardized SMILES or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get largest fragment (remove salts/counterions)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda x: x.GetNumAtoms())

    return Chem.MolToSmiles(mol, canonical=True)


def batch_validate_smiles(smiles_list: list[str]) -> list[tuple[str, bool, Optional[str]]]:
    """
    Validate a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of tuples (smiles, is_valid, error_message)
    """
    results = []
    for smiles in smiles_list:
        is_valid, error_msg = validate_smiles(smiles)
        results.append((smiles, is_valid, error_msg))
    return results


def smiles_to_inchi(smiles: str) -> Optional[str]:
    """
    Convert SMILES to InChI string.

    Args:
        smiles: SMILES string

    Returns:
        InChI string or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchi(mol)


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """
    Convert SMILES to InChIKey.

    Args:
        smiles: SMILES string

    Returns:
        InChIKey or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def calculate_drug_likeness(smiles: str) -> dict:
    """
    Calculate drug-likeness properties (Lipinski's Rule of Five).

    Args:
        smiles: SMILES string

    Returns:
        Dictionary with drug-likeness properties and violations
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    violations = []
    if mw > 500:
        violations.append("MW > 500")
    if logp > 5:
        violations.append("LogP > 5")
    if hbd > 5:
        violations.append("HBD > 5")
    if hba > 10:
        violations.append("HBA > 10")

    return {
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "lipinski_violations": len(violations),
        "violation_details": violations,
        "is_drug_like": len(violations) <= 1,
    }
