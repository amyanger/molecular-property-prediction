"""Tests for feature extraction utilities."""

import pytest
from rdkit import Chem

from src.utils.features import (
    one_hot,
    get_atom_features_gcn,
    get_atom_features_afp,
    get_bond_features,
    ATOM_FEATURES_GCN,
    ATOM_FEATURES_AFP,
    BOND_FEATURES,
)


class TestOneHot:
    """Tests for one-hot encoding function."""

    def test_one_hot_valid_value(self):
        """Test one-hot encoding with valid value."""
        choices = [1, 2, 3, 4, 5]
        result = one_hot(3, choices)
        assert result == [0, 0, 1, 0, 0]

    def test_one_hot_first_value(self):
        """Test one-hot encoding with first value."""
        choices = ["a", "b", "c"]
        result = one_hot("a", choices)
        assert result == [1, 0, 0]

    def test_one_hot_last_value(self):
        """Test one-hot encoding with last value."""
        choices = ["a", "b", "c"]
        result = one_hot("c", choices)
        assert result == [0, 0, 1]

    def test_one_hot_invalid_value(self):
        """Test one-hot encoding with value not in choices."""
        choices = [1, 2, 3]
        result = one_hot(99, choices)
        assert result == [0, 0, 0]

    def test_one_hot_empty_choices(self):
        """Test one-hot encoding with empty choices."""
        result = one_hot(1, [])
        assert result == []


class TestAtomFeaturesGCN:
    """Tests for GCN atom feature extraction."""

    def test_gcn_feature_dimension(self, simple_molecule_smiles):
        """Test that GCN features have correct dimension (141)."""
        mol = Chem.MolFromSmiles(simple_molecule_smiles)
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_gcn(atom)

        # Expected: 118 + 6 + 5 + 5 + 1 + 5 + 1 = 141
        assert len(features) == 141

    def test_gcn_features_carbon(self):
        """Test GCN features for carbon atom."""
        mol = Chem.MolFromSmiles("C")  # Methane
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_gcn(atom)

        # Check it's a list of numbers
        assert all(isinstance(f, (int, float)) for f in features)
        # Carbon is element 6, should have 1 in position 5 (0-indexed)
        assert features[5] == 1  # Atomic number = 6

    def test_gcn_features_aromatic(self):
        """Test GCN features correctly identify aromatic atoms."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_gcn(atom)

        # Aromatic flag should be 1
        # Position: 118 (atomic) + 6 (degree) + 5 (charge) + 5 (hybrid) = 134
        assert features[134] == 1

    def test_gcn_features_different_atoms(self):
        """Test that different atoms produce different features."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        carbon_features = get_atom_features_gcn(mol.GetAtomWithIdx(0))
        oxygen_features = get_atom_features_gcn(mol.GetAtomWithIdx(2))

        assert carbon_features != oxygen_features


class TestAtomFeaturesAFP:
    """Tests for AttentiveFP atom feature extraction."""

    def test_afp_feature_dimension(self, simple_molecule_smiles):
        """Test that AFP features have correct dimension (148)."""
        mol = Chem.MolFromSmiles(simple_molecule_smiles)
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_afp(atom)

        # Expected: 118 + 7 + 6 + 4 + 5 + 6 + 1 + 1 = 148
        assert len(features) == 148

    def test_afp_features_carbon(self):
        """Test AFP features for carbon atom."""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_afp(atom)

        assert all(isinstance(f, (int, float)) for f in features)
        assert features[5] == 1  # Carbon atomic number

    def test_afp_has_more_features_than_gcn(self, simple_molecule_smiles):
        """Test that AFP has more features than GCN."""
        mol = Chem.MolFromSmiles(simple_molecule_smiles)
        atom = mol.GetAtomWithIdx(0)

        gcn_features = get_atom_features_gcn(atom)
        afp_features = get_atom_features_afp(atom)

        assert len(afp_features) > len(gcn_features)


class TestBondFeatures:
    """Tests for bond feature extraction."""

    def test_bond_feature_dimension(self):
        """Test that bond features have correct dimension (12)."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Expected: 4 (bond type) + 6 (stereo) + 1 (conjugated) + 1 (ring) = 12
        assert len(features) == 12

    def test_single_bond_features(self):
        """Test features for single bond."""
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Single bond should have first position = 1
        assert features[0] == 1

    def test_double_bond_features(self):
        """Test features for double bond."""
        mol = Chem.MolFromSmiles("C=C")  # Ethene
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Double bond should have second position = 1
        assert features[1] == 1

    def test_aromatic_bond_features(self):
        """Test features for aromatic bond."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Aromatic bond should have fourth position = 1
        assert features[3] == 1

    def test_ring_bond_features(self):
        """Test that ring bonds are correctly identified."""
        mol = Chem.MolFromSmiles("C1CC1")  # Cyclopropane
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)

        # Ring flag is last position
        assert features[-1] == 1


class TestFeatureDictionaries:
    """Tests for feature definition dictionaries."""

    def test_gcn_atom_features_keys(self):
        """Test GCN atom features has expected keys."""
        expected_keys = ['atomic_num', 'degree', 'formal_charge',
                        'hybridization', 'is_aromatic', 'num_hs']
        for key in expected_keys:
            assert key in ATOM_FEATURES_GCN

    def test_afp_atom_features_keys(self):
        """Test AFP atom features has expected keys."""
        expected_keys = ['atomic_num', 'degree', 'formal_charge',
                        'chiral_tag', 'num_hs', 'hybridization']
        for key in expected_keys:
            assert key in ATOM_FEATURES_AFP

    def test_bond_features_keys(self):
        """Test bond features has expected keys."""
        expected_keys = ['bond_type', 'stereo']
        for key in expected_keys:
            assert key in BOND_FEATURES

    def test_atomic_num_range(self):
        """Test that atomic numbers cover periodic table."""
        assert len(ATOM_FEATURES_GCN['atomic_num']) == 118
        assert ATOM_FEATURES_GCN['atomic_num'][0] == 1  # Hydrogen
        assert ATOM_FEATURES_GCN['atomic_num'][-1] == 118  # Oganesson


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_atom_molecule(self):
        """Test feature extraction for single atom molecule."""
        mol = Chem.MolFromSmiles("[He]")  # Helium
        atom = mol.GetAtomWithIdx(0)

        gcn_features = get_atom_features_gcn(atom)
        afp_features = get_atom_features_afp(atom)

        assert len(gcn_features) == 141
        assert len(afp_features) == 148

    def test_charged_atom(self):
        """Test feature extraction for charged atom."""
        mol = Chem.MolFromSmiles("[NH4+]")  # Ammonium
        atom = mol.GetAtomWithIdx(0)

        features = get_atom_features_gcn(atom)
        assert len(features) == 141

    def test_complex_molecule(self, aspirin_smiles):
        """Test feature extraction for complex molecule."""
        mol = Chem.MolFromSmiles(aspirin_smiles)

        # Test all atoms
        for atom in mol.GetAtoms():
            gcn_features = get_atom_features_gcn(atom)
            afp_features = get_atom_features_afp(atom)
            assert len(gcn_features) == 141
            assert len(afp_features) == 148

        # Test all bonds
        for bond in mol.GetBonds():
            features = get_bond_features(bond)
            assert len(features) == 12
