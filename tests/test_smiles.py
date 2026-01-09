"""Tests for SMILES validation and processing utilities."""

import pytest

from src.utils.smiles import (
    validate_smiles,
    canonicalize_smiles,
    standardize_smiles,
    get_molecule_info,
    batch_validate_smiles,
    smiles_to_inchi,
    smiles_to_inchikey,
    calculate_drug_likeness,
)


class TestValidateSmiles:
    """Tests for SMILES validation."""

    def test_valid_smiles(self):
        """Test validation of valid SMILES."""
        is_valid, error = validate_smiles("CCO")
        assert is_valid is True
        assert error is None

    def test_valid_complex_smiles(self):
        """Test validation of complex SMILES."""
        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        is_valid, error = validate_smiles(aspirin)
        assert is_valid is True

    def test_invalid_smiles(self):
        """Test validation of invalid SMILES."""
        is_valid, error = validate_smiles("INVALID")
        assert is_valid is False
        assert error is not None

    def test_empty_smiles(self):
        """Test validation of empty SMILES."""
        is_valid, error = validate_smiles("")
        assert is_valid is False

    def test_none_smiles(self):
        """Test validation of None."""
        is_valid, error = validate_smiles(None)
        assert is_valid is False

    def test_whitespace_smiles(self):
        """Test validation of whitespace-only SMILES."""
        is_valid, error = validate_smiles("   ")
        assert is_valid is False


class TestCanonicalizeSmiles:
    """Tests for SMILES canonicalization."""

    def test_canonicalize_simple(self):
        """Test canonicalization of simple SMILES."""
        result = canonicalize_smiles("C(C)O")
        assert result == "CCO"

    def test_canonicalize_aromatic(self):
        """Test canonicalization of aromatic compounds."""
        result = canonicalize_smiles("c1ccccc1")
        assert result is not None
        assert "c" in result.lower()  # Should contain aromatic notation

    def test_canonicalize_invalid(self):
        """Test canonicalization of invalid SMILES."""
        result = canonicalize_smiles("INVALID")
        assert result is None


class TestStandardizeSmiles:
    """Tests for SMILES standardization."""

    def test_standardize_salt(self):
        """Test standardization removes salt."""
        # Sodium acetate
        result = standardize_smiles("CC(=O)[O-].[Na+]")
        assert result is not None
        # Should only contain the main fragment
        assert "." not in result

    def test_standardize_simple(self):
        """Test standardization of simple molecule."""
        result = standardize_smiles("CCO")
        assert result == "CCO"


class TestGetMoleculeInfo:
    """Tests for molecule information extraction."""

    def test_molecule_info_ethanol(self):
        """Test info extraction for ethanol."""
        info = get_molecule_info("CCO")

        assert info.is_valid is True
        assert info.num_atoms == 3
        assert info.num_heavy_atoms == 2
        assert info.num_bonds == 2
        assert info.molecular_weight > 40 and info.molecular_weight < 50  # ~46

    def test_molecule_info_benzene(self):
        """Test info extraction for benzene."""
        info = get_molecule_info("c1ccccc1")

        assert info.is_valid is True
        assert info.num_aromatic_rings == 1
        assert info.num_rings == 1

    def test_molecule_info_aspirin(self):
        """Test info extraction for aspirin."""
        info = get_molecule_info("CC(=O)OC1=CC=CC=C1C(=O)O")

        assert info.is_valid is True
        assert info.molecular_weight > 175 and info.molecular_weight < 185  # ~180

    def test_molecule_info_invalid(self):
        """Test info extraction for invalid SMILES."""
        info = get_molecule_info("INVALID")

        assert info.is_valid is False
        assert info.error_message is not None
        assert info.num_atoms == 0


class TestBatchValidateSmiles:
    """Tests for batch SMILES validation."""

    def test_batch_validate_mixed(self):
        """Test batch validation with mixed valid/invalid."""
        smiles_list = ["CCO", "INVALID", "c1ccccc1", ""]

        results = batch_validate_smiles(smiles_list)

        assert len(results) == 4
        assert results[0][1] is True   # CCO is valid
        assert results[1][1] is False  # INVALID is invalid
        assert results[2][1] is True   # benzene is valid
        assert results[3][1] is False  # empty is invalid

    def test_batch_validate_all_valid(self):
        """Test batch validation with all valid."""
        smiles_list = ["CCO", "C", "CC"]

        results = batch_validate_smiles(smiles_list)

        assert all(r[1] for r in results)


class TestSmilesToInchi:
    """Tests for SMILES to InChI conversion."""

    def test_smiles_to_inchi(self):
        """Test InChI generation."""
        inchi = smiles_to_inchi("CCO")

        assert inchi is not None
        assert inchi.startswith("InChI=")

    def test_smiles_to_inchi_invalid(self):
        """Test InChI for invalid SMILES."""
        inchi = smiles_to_inchi("INVALID")
        assert inchi is None


class TestSmilesToInchikey:
    """Tests for SMILES to InChIKey conversion."""

    def test_smiles_to_inchikey(self):
        """Test InChIKey generation."""
        inchikey = smiles_to_inchikey("CCO")

        assert inchikey is not None
        assert len(inchikey) == 27  # Standard InChIKey length

    def test_smiles_to_inchikey_invalid(self):
        """Test InChIKey for invalid SMILES."""
        inchikey = smiles_to_inchikey("INVALID")
        assert inchikey is None


class TestDrugLikeness:
    """Tests for drug-likeness calculation."""

    def test_drug_likeness_aspirin(self):
        """Test drug-likeness for aspirin (should pass Lipinski)."""
        result = calculate_drug_likeness("CC(=O)OC1=CC=CC=C1C(=O)O")

        assert "error" not in result
        assert result["is_drug_like"] is True
        assert result["lipinski_violations"] <= 1

    def test_drug_likeness_large_molecule(self):
        """Test drug-likeness for a large molecule."""
        # Create a molecule that violates Lipinski's rules
        # High MW, high LogP
        large_mol = "CCCCCCCCCCCCCCCCCCCCCCCCCC"  # Long chain
        result = calculate_drug_likeness(large_mol)

        assert "molecular_weight" in result
        assert "logp" in result

    def test_drug_likeness_invalid(self):
        """Test drug-likeness for invalid SMILES."""
        result = calculate_drug_likeness("INVALID")

        assert "error" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_atom(self):
        """Test single atom molecules."""
        info = get_molecule_info("C")

        assert info.is_valid is True
        assert info.num_atoms == 1

    def test_charged_molecule(self):
        """Test charged molecules."""
        is_valid, _ = validate_smiles("[NH4+]")
        assert is_valid is True

    def test_radical(self):
        """Test molecules with radicals."""
        is_valid, _ = validate_smiles("[CH3]")
        assert is_valid is True

    def test_unicode_handling(self):
        """Test that unicode doesn't crash validation."""
        is_valid, _ = validate_smiles("CCO\u200b")  # Zero-width space
        # Should handle gracefully (either valid or proper error)
        assert isinstance(is_valid, bool)
