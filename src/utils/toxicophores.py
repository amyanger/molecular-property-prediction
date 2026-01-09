"""Toxicophore (toxic structural alert) detection utilities."""

from typing import Optional
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToxicophoreMatch:
    """Container for a single toxicophore match."""

    name: str
    smarts: str
    atom_indices: tuple[int, ...]
    description: str
    severity: str  # "high", "medium", "low"


@dataclass
class ToxicophoreResult:
    """Container for toxicophore detection results."""

    smiles: str
    total_alerts: int
    high_severity: int
    medium_severity: int
    low_severity: int
    matches: list[ToxicophoreMatch] = field(default_factory=list)
    is_safe: bool = True


# Toxicophore SMARTS patterns with descriptions and severity
TOXICOPHORE_PATTERNS = {
    # Reactive/Electrophilic groups
    "acyl_halide": {
        "smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
        "description": "Acyl halide - highly reactive, causes tissue damage",
        "severity": "high",
    },
    "acid_anhydride": {
        "smarts": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "description": "Acid anhydride - reactive, irritant",
        "severity": "high",
    },
    "epoxide": {
        "smarts": "C1OC1",
        "description": "Epoxide - DNA alkylating agent, mutagenic",
        "severity": "high",
    },
    "aziridine": {
        "smarts": "C1NC1",
        "description": "Aziridine - DNA alkylating agent, mutagenic",
        "severity": "high",
    },
    "michael_acceptor": {
        "smarts": "[CX3]=[CX3][CX3](=[OX1])",
        "description": "Michael acceptor - protein reactive, cytotoxic",
        "severity": "high",
    },
    "aldehyde": {
        "smarts": "[CX3H1](=[OX1])",
        "description": "Aldehyde - reactive, forms Schiff bases",
        "severity": "medium",
    },
    "isocyanate": {
        "smarts": "[NX2]=[CX2]=[OX1]",
        "description": "Isocyanate - respiratory sensitizer",
        "severity": "high",
    },
    "isothiocyanate": {
        "smarts": "[NX2]=[CX2]=[SX1]",
        "description": "Isothiocyanate - reactive, irritant",
        "severity": "medium",
    },

    # Genotoxic/Mutagenic
    "nitro_aromatic": {
        "smarts": "c[N+](=O)[O-]",
        "description": "Nitro aromatic - mutagenic, carcinogenic",
        "severity": "high",
    },
    "nitroso": {
        "smarts": "[NX2]=[OX1]",
        "description": "Nitroso compound - mutagenic",
        "severity": "high",
    },
    "n_oxide": {
        "smarts": "[nX3+]([O-])",
        "description": "Aromatic N-oxide - DNA intercalation",
        "severity": "medium",
    },
    "azo": {
        "smarts": "[NX2]=[NX2]",
        "description": "Azo compound - potential carcinogen via reduction",
        "severity": "medium",
    },
    "hydrazine": {
        "smarts": "[NX3][NX3]",
        "description": "Hydrazine - hepatotoxic, carcinogenic",
        "severity": "high",
    },
    "triazene": {
        "smarts": "[NX2]=[NX2][NX3]",
        "description": "Triazene - DNA alkylating agent",
        "severity": "high",
    },
    "n_nitroso": {
        "smarts": "[NX3][NX2]=[OX1]",
        "description": "N-nitroso - potent carcinogen",
        "severity": "high",
    },

    # Heavy atom containing
    "organomercury": {
        "smarts": "[Hg]",
        "description": "Organomercury - neurotoxic",
        "severity": "high",
    },
    "organolead": {
        "smarts": "[Pb]",
        "description": "Organolead - neurotoxic, developmental toxicity",
        "severity": "high",
    },
    "organoarsenic": {
        "smarts": "[As]",
        "description": "Organoarsenic - carcinogenic",
        "severity": "high",
    },
    "organocadmium": {
        "smarts": "[Cd]",
        "description": "Organocadmium - nephrotoxic, carcinogenic",
        "severity": "high",
    },
    "organothallium": {
        "smarts": "[Tl]",
        "description": "Organothallium - highly toxic",
        "severity": "high",
    },

    # Halogenated compounds
    "polychlorinated_biphenyl": {
        "smarts": "c1ccc(Cl)cc1-c2ccc(Cl)cc2",
        "description": "Polychlorinated biphenyl (PCB) - persistent, endocrine disruptor",
        "severity": "high",
    },
    "halogenated_alkene": {
        "smarts": "[F,Cl,Br,I][CX3]=[CX3]",
        "description": "Halogenated alkene - reactive, mutagenic",
        "severity": "medium",
    },
    "perhaloketone": {
        "smarts": "[CX3](=[OX1])([F,Cl])([F,Cl])",
        "description": "Perhaloketone - highly reactive",
        "severity": "high",
    },

    # Sulfur compounds
    "thiourea": {
        "smarts": "[NX3][CX3](=[SX1])[NX3]",
        "description": "Thiourea - thyroid disruptor",
        "severity": "medium",
    },
    "sulfonic_acid_ester": {
        "smarts": "[SX4](=[OX1])(=[OX1])([OX2])",
        "description": "Sulfonic acid ester - DNA alkylating agent",
        "severity": "high",
    },
    "thioamide": {
        "smarts": "[NX3][CX3]=[SX1]",
        "description": "Thioamide - hepatotoxic",
        "severity": "medium",
    },

    # Nitrogen compounds
    "quaternary_nitrogen": {
        "smarts": "[N+;!$([N+]=[*]);!$([N+](=O)O)]",
        "description": "Quaternary nitrogen - potential neurotoxicity",
        "severity": "low",
    },
    "primary_aromatic_amine": {
        "smarts": "[cX3][NX3H2]",
        "description": "Primary aromatic amine - carcinogenic metabolites",
        "severity": "medium",
    },
    "secondary_aromatic_amine": {
        "smarts": "[cX3][NX3H1][CX4]",
        "description": "Secondary aromatic amine - potential carcinogen",
        "severity": "medium",
    },
    "polycyclic_aromatic": {
        "smarts": "c1ccc2c(c1)ccc3ccccc32",
        "description": "Polycyclic aromatic hydrocarbon - carcinogenic",
        "severity": "medium",
    },

    # Phosphorus compounds
    "phosphoramide": {
        "smarts": "[PX4](=[OX1])([NX3])([NX3])",
        "description": "Phosphoramide - DNA crosslinker",
        "severity": "high",
    },
    "organophosphate": {
        "smarts": "[PX4](=[OX1])([OX2])([OX2])",
        "description": "Organophosphate - cholinesterase inhibitor",
        "severity": "medium",
    },

    # Other alerts
    "peroxide": {
        "smarts": "[OX2][OX2]",
        "description": "Peroxide - oxidative stress, explosion hazard",
        "severity": "medium",
    },
    "cyanide": {
        "smarts": "[C-]#[N+]",
        "description": "Cyanide - cytochrome oxidase inhibitor",
        "severity": "high",
    },
    "azide": {
        "smarts": "[N-]=[N+]=[N-]",
        "description": "Azide - cytochrome oxidase inhibitor",
        "severity": "high",
    },
    "disulfide": {
        "smarts": "[SX2][SX2]",
        "description": "Disulfide - protein reactive",
        "severity": "low",
    },
}


class ToxicophoreDetector:
    """
    Detect toxic structural alerts (toxicophores) in molecules.

    Identifies potentially toxic substructures based on known
    structural alerts associated with various toxicity endpoints.
    """

    def __init__(self, include_rdkit_filters: bool = True):
        """
        Initialize toxicophore detector.

        Args:
            include_rdkit_filters: Whether to include RDKit's built-in filters
        """
        # Compile SMARTS patterns
        self.patterns = {}
        for name, info in TOXICOPHORE_PATTERNS.items():
            pattern = Chem.MolFromSmarts(info["smarts"])
            if pattern is not None:
                self.patterns[name] = {
                    "pattern": pattern,
                    "smarts": info["smarts"],
                    "description": info["description"],
                    "severity": info["severity"],
                }
            else:
                logger.warning(f"Invalid SMARTS pattern for {name}: {info['smarts']}")

        # RDKit filter catalogs
        self.include_rdkit = include_rdkit_filters
        if include_rdkit_filters:
            self._init_rdkit_catalogs()

    def _init_rdkit_catalogs(self):
        """Initialize RDKit filter catalogs."""
        # PAINS filters
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(params_pains)

        # Brenk filters (unwanted substructures)
        params_brenk = FilterCatalogParams()
        params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog.FilterCatalog(params_brenk)

    def detect(self, smiles: str) -> ToxicophoreResult:
        """
        Detect toxicophores in a molecule.

        Args:
            smiles: SMILES string

        Returns:
            ToxicophoreResult object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToxicophoreResult(
                smiles=smiles,
                total_alerts=0,
                high_severity=0,
                medium_severity=0,
                low_severity=0,
                is_safe=False,  # Invalid molecule
            )

        matches = []
        severity_counts = {"high": 0, "medium": 0, "low": 0}

        # Check custom toxicophore patterns
        for name, info in self.patterns.items():
            pattern_matches = mol.GetSubstructMatches(info["pattern"])
            for match_atoms in pattern_matches:
                matches.append(
                    ToxicophoreMatch(
                        name=name,
                        smarts=info["smarts"],
                        atom_indices=match_atoms,
                        description=info["description"],
                        severity=info["severity"],
                    )
                )
                severity_counts[info["severity"]] += 1

        # Check RDKit filters
        if self.include_rdkit:
            # PAINS
            pains_matches = self.pains_catalog.GetMatches(mol)
            for entry in pains_matches:
                matches.append(
                    ToxicophoreMatch(
                        name=f"PAINS: {entry.GetDescription()}",
                        smarts="",
                        atom_indices=tuple(),
                        description="PAINS (Pan-Assay Interference) compound",
                        severity="medium",
                    )
                )
                severity_counts["medium"] += 1

            # Brenk
            brenk_matches = self.brenk_catalog.GetMatches(mol)
            for entry in brenk_matches:
                matches.append(
                    ToxicophoreMatch(
                        name=f"Brenk: {entry.GetDescription()}",
                        smarts="",
                        atom_indices=tuple(),
                        description="Brenk structural alert",
                        severity="medium",
                    )
                )
                severity_counts["medium"] += 1

        total_alerts = len(matches)
        is_safe = severity_counts["high"] == 0

        return ToxicophoreResult(
            smiles=smiles,
            total_alerts=total_alerts,
            high_severity=severity_counts["high"],
            medium_severity=severity_counts["medium"],
            low_severity=severity_counts["low"],
            matches=matches,
            is_safe=is_safe,
        )

    def filter_safe_molecules(
        self,
        smiles_list: list[str],
        max_high_severity: int = 0,
        max_total_alerts: int = 2,
    ) -> list[str]:
        """
        Filter molecules to keep only those without serious toxicophores.

        Args:
            smiles_list: List of SMILES strings
            max_high_severity: Maximum allowed high severity alerts
            max_total_alerts: Maximum allowed total alerts

        Returns:
            List of safe SMILES
        """
        safe_molecules = []

        for smiles in smiles_list:
            result = self.detect(smiles)
            if result.high_severity <= max_high_severity and result.total_alerts <= max_total_alerts:
                safe_molecules.append(smiles)

        logger.info(f"Filtered {len(smiles_list)} -> {len(safe_molecules)} safe molecules")
        return safe_molecules

    def get_pattern_names(self) -> list[str]:
        """Get list of all toxicophore pattern names."""
        return list(self.patterns.keys())


def quick_toxicophore_check(smiles: str) -> dict:
    """
    Quick check for common toxicophores.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary with toxicophore check results
    """
    detector = ToxicophoreDetector(include_rdkit_filters=False)
    result = detector.detect(smiles)

    return {
        "smiles": smiles,
        "is_safe": result.is_safe,
        "total_alerts": result.total_alerts,
        "high_severity_alerts": result.high_severity,
        "alert_names": [m.name for m in result.matches],
    }


def highlight_toxicophores(smiles: str) -> Optional[str]:
    """
    Generate SVG image with highlighted toxicophores.

    Args:
        smiles: SMILES string

    Returns:
        SVG string or None if no toxicophores found
    """
    from rdkit.Chem import Draw

    detector = ToxicophoreDetector(include_rdkit_filters=False)
    result = detector.detect(smiles)

    if result.total_alerts == 0:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Collect all atom indices to highlight
    highlight_atoms = set()
    for match in result.matches:
        highlight_atoms.update(match.atom_indices)

    # Generate SVG
    drawer = Draw.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms))
    drawer.FinishDrawing()

    return drawer.GetDrawingText()
