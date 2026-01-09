"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Sample SMILES strings for testing
SAMPLE_SMILES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "methane": "C",
    "water": "O",
    "invalid": "INVALID_SMILES_STRING",
}


@pytest.fixture
def sample_smiles():
    """Provide sample SMILES strings for testing."""
    return SAMPLE_SMILES.copy()


@pytest.fixture
def aspirin_smiles():
    """Provide aspirin SMILES string."""
    return SAMPLE_SMILES["aspirin"]


@pytest.fixture
def simple_molecule_smiles():
    """Provide a simple molecule (ethanol) for basic tests."""
    return SAMPLE_SMILES["ethanol"]


@pytest.fixture
def invalid_smiles():
    """Provide an invalid SMILES string."""
    return SAMPLE_SMILES["invalid"]
