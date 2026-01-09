"""Molecular visualization utilities."""

import io
import base64
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_image(
    smiles: str,
    size: tuple[int, int] = (300, 300),
    highlight_atoms: Optional[list[int]] = None,
    highlight_bonds: Optional[list[int]] = None,
) -> Optional[bytes]:
    """
    Convert SMILES to PNG image bytes.

    Args:
        smiles: SMILES string
        size: Image size as (width, height)
        highlight_atoms: List of atom indices to highlight
        highlight_bonds: List of bond indices to highlight

    Returns:
        PNG image as bytes or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    # Draw options
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True

    # Draw molecule
    if highlight_atoms or highlight_bonds:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms or [],
            highlightBonds=highlight_bonds or [],
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_svg(
    smiles: str,
    size: tuple[int, int] = (300, 300),
    highlight_atoms: Optional[list[int]] = None,
) -> Optional[str]:
    """
    Convert SMILES to SVG string.

    Args:
        smiles: SMILES string
        size: Image size as (width, height)
        highlight_atoms: List of atom indices to highlight

    Returns:
        SVG string or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])

    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_base64(smiles: str, size: tuple[int, int] = (300, 300)) -> Optional[str]:
    """
    Convert SMILES to base64-encoded PNG for HTML embedding.

    Args:
        smiles: SMILES string
        size: Image size

    Returns:
        Base64 encoded string or None if invalid
    """
    img_bytes = mol_to_image(smiles, size)
    if img_bytes is None:
        return None
    return base64.b64encode(img_bytes).decode('utf-8')


def mol_to_html_img(smiles: str, size: tuple[int, int] = (300, 300)) -> Optional[str]:
    """
    Convert SMILES to HTML img tag with embedded base64 image.

    Args:
        smiles: SMILES string
        size: Image size

    Returns:
        HTML img tag string or None if invalid
    """
    b64 = mol_to_base64(smiles, size)
    if b64 is None:
        return None
    return f'<img src="data:image/png;base64,{b64}" width="{size[0]}" height="{size[1]}" />'


def draw_molecule_grid(
    smiles_list: list[str],
    legends: Optional[list[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 200),
) -> Optional[bytes]:
    """
    Draw a grid of molecules.

    Args:
        smiles_list: List of SMILES strings
        legends: Optional list of labels for each molecule
        mols_per_row: Number of molecules per row
        sub_img_size: Size of each sub-image

    Returns:
        PNG image as bytes or None if all invalid
    """
    mols = []
    valid_legends = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            if legends:
                valid_legends.append(legends[i] if i < len(legends) else "")
            else:
                valid_legends.append(smi[:20] + "..." if len(smi) > 20 else smi)

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=valid_legends,
    )

    # Convert PIL image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


def highlight_substructure(
    smiles: str,
    substructure_smarts: str,
    size: tuple[int, int] = (300, 300),
) -> Optional[bytes]:
    """
    Highlight a substructure in a molecule.

    Args:
        smiles: SMILES string of the molecule
        substructure_smarts: SMARTS pattern to highlight
        size: Image size

    Returns:
        PNG image with highlighted substructure or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pattern = Chem.MolFromSmarts(substructure_smarts)
    if pattern is None:
        return mol_to_image(smiles, size)

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return mol_to_image(smiles, size)

    # Get all matching atoms
    highlight_atoms = list(set(atom for match in matches for atom in match))

    return mol_to_image(smiles, size, highlight_atoms=highlight_atoms)


def draw_similarity_map(
    smiles: str,
    weights: list[float],
    size: tuple[int, int] = (300, 300),
) -> Optional[bytes]:
    """
    Draw a molecule with atom-level similarity/importance weights.

    Args:
        smiles: SMILES string
        weights: List of weights for each atom (same order as atoms)
        size: Image size

    Returns:
        PNG image with colored atoms based on weights
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if len(weights) != mol.GetNumAtoms():
        return mol_to_image(smiles, size)

    AllChem.Compute2DCoords(mol)

    # Use SimilarityMaps for visualization
    from rdkit.Chem.Draw import SimilarityMaps

    img_bytes = io.BytesIO()
    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights,
        size=size,
    )
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    return img_bytes.getvalue()
