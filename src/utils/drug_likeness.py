"""Drug-likeness and synthetic accessibility utilities."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrugLikenessResult:
    """Container for drug-likeness analysis results."""

    smiles: str
    molecular_weight: float
    logp: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    lipinski_violations: int
    veber_violations: int
    is_lipinski_compliant: bool
    is_veber_compliant: bool
    is_drug_like: bool
    qed_score: float  # Quantitative estimate of drug-likeness
    synthetic_accessibility: float


@dataclass
class ADMEProperties:
    """Container for ADME (Absorption, Distribution, Metabolism, Excretion) properties."""

    smiles: str
    # Absorption
    gi_absorption: str  # High/Low
    bioavailability_score: float

    # Distribution
    bbb_permeant: bool  # Blood-brain barrier
    pgp_substrate: bool  # P-glycoprotein

    # Metabolism (CYP inhibition)
    cyp1a2_inhibitor: bool
    cyp2c19_inhibitor: bool
    cyp2c9_inhibitor: bool
    cyp2d6_inhibitor: bool
    cyp3a4_inhibitor: bool

    # Alerts
    pains_alerts: int
    brenk_alerts: int


def compute_synthetic_accessibility(mol: Chem.Mol) -> float:
    """
    Compute synthetic accessibility score (SA Score).

    Based on: Ertl & Schuffenhauer, J. Cheminformatics 2009

    Score ranges from 1 (easy to synthesize) to 10 (difficult).

    Args:
        mol: RDKit molecule object

    Returns:
        SA score (1-10)
    """
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer

    try:
        sa_score = sascorer.calculateScore(mol)
        return sa_score
    except Exception:
        # Fallback: estimate based on complexity
        return _estimate_sa_score(mol)


def _estimate_sa_score(mol: Chem.Mol) -> float:
    """Estimate SA score using molecular descriptors."""
    # Get descriptors
    mw = Descriptors.MolWt(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol))
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_hba = rdMolDescriptors.CalcNumHBA(mol)
    n_hbd = rdMolDescriptors.CalcNumHBD(mol)

    # Estimate SA score
    sa = 1.0

    # Penalize high molecular weight
    if mw > 500:
        sa += (mw - 500) / 100

    # Penalize many rings
    sa += n_rings * 0.3

    # Penalize stereocenters
    sa += n_stereo * 0.5

    # Penalize flexibility
    if n_rotatable > 10:
        sa += (n_rotatable - 10) * 0.1

    # Normalize to 1-10 range
    sa = min(max(sa, 1.0), 10.0)

    return sa


def compute_qed_score(mol: Chem.Mol) -> float:
    """
    Compute Quantitative Estimate of Drug-likeness (QED).

    Based on: Bickerton et al., Nature Chemistry 2012

    Score ranges from 0 (not drug-like) to 1 (very drug-like).

    Args:
        mol: RDKit molecule object

    Returns:
        QED score (0-1)
    """
    from rdkit.Chem.QED import qed
    return qed(mol)


def check_lipinski_rule(mol: Chem.Mol) -> Tuple[int, dict]:
    """
    Check Lipinski's Rule of Five violations.

    Rules:
    - MW <= 500
    - LogP <= 5
    - HBD <= 5
    - HBA <= 10

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (number of violations, detailed results)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    violations = 0
    results = {
        "molecular_weight": {"value": mw, "threshold": 500, "passed": mw <= 500},
        "logp": {"value": logp, "threshold": 5, "passed": logp <= 5},
        "hbd": {"value": hbd, "threshold": 5, "passed": hbd <= 5},
        "hba": {"value": hba, "threshold": 10, "passed": hba <= 10},
    }

    for rule in results.values():
        if not rule["passed"]:
            violations += 1

    return violations, results


def check_veber_rule(mol: Chem.Mol) -> Tuple[int, dict]:
    """
    Check Veber's rules for oral bioavailability.

    Rules:
    - TPSA <= 140 Å²
    - Rotatable bonds <= 10

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (number of violations, detailed results)
    """
    tpsa = Descriptors.TPSA(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

    violations = 0
    results = {
        "tpsa": {"value": tpsa, "threshold": 140, "passed": tpsa <= 140},
        "rotatable_bonds": {"value": n_rotatable, "threshold": 10, "passed": n_rotatable <= 10},
    }

    for rule in results.values():
        if not rule["passed"]:
            violations += 1

    return violations, results


def analyze_drug_likeness(smiles: str) -> Optional[DrugLikenessResult]:
    """
    Perform comprehensive drug-likeness analysis.

    Args:
        smiles: SMILES string

    Returns:
        DrugLikenessResult or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Check rules
    lipinski_violations, _ = check_lipinski_rule(mol)
    veber_violations, _ = check_veber_rule(mol)

    # Compute scores
    qed = compute_qed_score(mol)
    sa = compute_synthetic_accessibility(mol)

    return DrugLikenessResult(
        smiles=smiles,
        molecular_weight=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        rotatable_bonds=n_rotatable,
        lipinski_violations=lipinski_violations,
        veber_violations=veber_violations,
        is_lipinski_compliant=lipinski_violations <= 1,
        is_veber_compliant=veber_violations == 0,
        is_drug_like=lipinski_violations <= 1 and veber_violations == 0,
        qed_score=qed,
        synthetic_accessibility=sa,
    )


def predict_gi_absorption(mol: Chem.Mol) -> str:
    """
    Predict gastrointestinal absorption.

    Based on the BOILED-Egg model.

    Args:
        mol: RDKit molecule object

    Returns:
        "High" or "Low"
    """
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    # BOILED-Egg white region (HIA)
    if tpsa <= 131.6 and logp >= -2.3 and logp <= 6.8:
        return "High"
    return "Low"


def predict_bbb_permeation(mol: Chem.Mol) -> bool:
    """
    Predict blood-brain barrier permeation.

    Based on the BOILED-Egg model.

    Args:
        mol: RDKit molecule object

    Returns:
        True if BBB permeant
    """
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    # BOILED-Egg yolk region (BBB)
    if tpsa <= 79.0 and logp >= 0.4 and logp <= 6.0:
        return True
    return False


def compute_bioavailability_score(mol: Chem.Mol) -> float:
    """
    Compute bioavailability score.

    Based on Martin's criteria and extended rules.

    Args:
        mol: RDKit molecule object

    Returns:
        Bioavailability score (0-1)
    """
    tpsa = Descriptors.TPSA(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    mw = Descriptors.MolWt(mol)

    score = 0.0

    # TPSA rule
    if tpsa <= 140:
        score += 0.25

    # Rotatable bonds
    if n_rotatable <= 10:
        score += 0.25

    # HBD
    if hbd <= 5:
        score += 0.25

    # MW
    if mw <= 500:
        score += 0.25

    return score


def analyze_adme_properties(smiles: str) -> Optional[ADMEProperties]:
    """
    Analyze ADME properties of a molecule.

    Args:
        smiles: SMILES string

    Returns:
        ADMEProperties or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Absorption
    gi_absorption = predict_gi_absorption(mol)
    bioavailability = compute_bioavailability_score(mol)

    # Distribution
    bbb = predict_bbb_permeation(mol)
    pgp = _predict_pgp_substrate(mol)

    # Metabolism - CYP inhibition (simplified model)
    cyp1a2 = _predict_cyp_inhibition(mol, "1a2")
    cyp2c19 = _predict_cyp_inhibition(mol, "2c19")
    cyp2c9 = _predict_cyp_inhibition(mol, "2c9")
    cyp2d6 = _predict_cyp_inhibition(mol, "2d6")
    cyp3a4 = _predict_cyp_inhibition(mol, "3a4")

    # Alerts
    pains = count_pains_alerts(mol)
    brenk = count_brenk_alerts(mol)

    return ADMEProperties(
        smiles=smiles,
        gi_absorption=gi_absorption,
        bioavailability_score=bioavailability,
        bbb_permeant=bbb,
        pgp_substrate=pgp,
        cyp1a2_inhibitor=cyp1a2,
        cyp2c19_inhibitor=cyp2c19,
        cyp2c9_inhibitor=cyp2c9,
        cyp2d6_inhibitor=cyp2d6,
        cyp3a4_inhibitor=cyp3a4,
        pains_alerts=pains,
        brenk_alerts=brenk,
    )


def _predict_pgp_substrate(mol: Chem.Mol) -> bool:
    """Predict P-glycoprotein substrate (simplified model)."""
    mw = Descriptors.MolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)

    # P-gp substrates tend to have MW > 400 and HBD > 4
    return mw > 400 and hbd > 4


def _predict_cyp_inhibition(mol: Chem.Mol, cyp_type: str) -> bool:
    """Predict CYP inhibition (simplified model based on descriptors)."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

    # CYP inhibitors tend to be lipophilic with aromatic rings
    if cyp_type in ["1a2", "2d6"]:
        return logp > 2 and n_aromatic >= 2
    elif cyp_type in ["2c9", "2c19"]:
        return logp > 3 and mw > 300
    else:  # 3a4
        return mw > 400 and logp > 3


def count_pains_alerts(mol: Chem.Mol) -> int:
    """
    Count PAINS (Pan-Assay Interference Compounds) alerts.

    Args:
        mol: RDKit molecule object

    Returns:
        Number of PAINS alerts
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    return len(catalog.GetMatches(mol))


def count_brenk_alerts(mol: Chem.Mol) -> int:
    """
    Count Brenk structural alerts (unwanted substructures).

    Args:
        mol: RDKit molecule object

    Returns:
        Number of Brenk alerts
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)

    return len(catalog.GetMatches(mol))


def filter_drug_like_molecules(
    smiles_list: list[str],
    max_lipinski_violations: int = 1,
    require_veber: bool = True,
    min_qed: float = 0.3,
    max_sa_score: float = 6.0,
    max_pains_alerts: int = 0,
) -> list[str]:
    """
    Filter molecules by drug-likeness criteria.

    Args:
        smiles_list: List of SMILES strings
        max_lipinski_violations: Maximum allowed Lipinski violations
        require_veber: Whether to require Veber compliance
        min_qed: Minimum QED score
        max_sa_score: Maximum SA score
        max_pains_alerts: Maximum allowed PAINS alerts

    Returns:
        List of filtered SMILES
    """
    filtered = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Check Lipinski
        lipinski_violations, _ = check_lipinski_rule(mol)
        if lipinski_violations > max_lipinski_violations:
            continue

        # Check Veber
        if require_veber:
            veber_violations, _ = check_veber_rule(mol)
            if veber_violations > 0:
                continue

        # Check QED
        qed = compute_qed_score(mol)
        if qed < min_qed:
            continue

        # Check SA score
        sa = compute_synthetic_accessibility(mol)
        if sa > max_sa_score:
            continue

        # Check PAINS
        pains = count_pains_alerts(mol)
        if pains > max_pains_alerts:
            continue

        filtered.append(smiles)

    logger.info(f"Filtered {len(smiles_list)} -> {len(filtered)} drug-like molecules")
    return filtered
