"""Chemical space visualization using dimensionality reduction."""

import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChemicalSpaceEmbedding:
    """Container for chemical space embedding results."""

    coordinates: np.ndarray  # (N, 2) or (N, 3)
    smiles_list: list[str]
    method: str
    fingerprint_type: str
    labels: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


class ChemicalSpaceVisualizer:
    """
    Visualize chemical space using dimensionality reduction.

    Converts molecules to fingerprints and projects to 2D/3D using
    t-SNE, PCA, or UMAP.

    Args:
        fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'maccs')
        radius: Radius for Morgan fingerprints
        n_bits: Number of bits for fingerprints
    """

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits

    def _get_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Generate fingerprint as numpy array."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.fingerprint_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        elif self.fingerprint_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        elif self.fingerprint_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

        arr = np.zeros(fp.GetNumBits())
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def compute_fingerprints(
        self,
        smiles_list: list[str],
    ) -> Tuple[np.ndarray, list[str], list[int]]:
        """
        Compute fingerprints for list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of (fingerprint_matrix, valid_smiles, valid_indices)
        """
        fingerprints = []
        valid_smiles = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            fp = self._get_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_smiles.append(smiles)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")

        return np.array(fingerprints), valid_smiles, valid_indices

    def embed_tsne(
        self,
        smiles_list: list[str],
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        random_state: int = 42,
        labels: Optional[np.ndarray] = None,
    ) -> ChemicalSpaceEmbedding:
        """
        Embed molecules using t-SNE.

        Args:
            smiles_list: List of SMILES strings
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            learning_rate: Learning rate ('auto' or float)
            n_iter: Number of iterations
            random_state: Random seed
            labels: Optional labels for molecules

        Returns:
            ChemicalSpaceEmbedding object
        """
        fps, valid_smiles, valid_indices = self.compute_fingerprints(smiles_list)

        if len(fps) == 0:
            raise ValueError("No valid molecules found")

        # Adjust perplexity if needed
        effective_perplexity = min(perplexity, len(fps) - 1)

        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
            metric="jaccard",  # Better for binary fingerprints
        )

        coordinates = tsne.fit_transform(fps)

        # Filter labels if provided
        valid_labels = None
        if labels is not None:
            valid_labels = labels[valid_indices]

        return ChemicalSpaceEmbedding(
            coordinates=coordinates,
            smiles_list=valid_smiles,
            method="t-SNE",
            fingerprint_type=self.fingerprint_type,
            labels=valid_labels,
            metadata={
                "perplexity": effective_perplexity,
                "n_iter": n_iter,
                "kl_divergence": float(tsne.kl_divergence_) if hasattr(tsne, 'kl_divergence_') else None,
            },
        )

    def embed_pca(
        self,
        smiles_list: list[str],
        n_components: int = 2,
        labels: Optional[np.ndarray] = None,
    ) -> ChemicalSpaceEmbedding:
        """
        Embed molecules using PCA.

        Args:
            smiles_list: List of SMILES strings
            n_components: Number of dimensions
            labels: Optional labels for molecules

        Returns:
            ChemicalSpaceEmbedding object
        """
        fps, valid_smiles, valid_indices = self.compute_fingerprints(smiles_list)

        if len(fps) == 0:
            raise ValueError("No valid molecules found")

        pca = PCA(n_components=n_components)
        coordinates = pca.fit_transform(fps)

        valid_labels = None
        if labels is not None:
            valid_labels = labels[valid_indices]

        return ChemicalSpaceEmbedding(
            coordinates=coordinates,
            smiles_list=valid_smiles,
            method="PCA",
            fingerprint_type=self.fingerprint_type,
            labels=valid_labels,
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            },
        )

    def embed_umap(
        self,
        smiles_list: list[str],
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "jaccard",
        random_state: int = 42,
        labels: Optional[np.ndarray] = None,
    ) -> ChemicalSpaceEmbedding:
        """
        Embed molecules using UMAP.

        Args:
            smiles_list: List of SMILES strings
            n_components: Number of dimensions
            n_neighbors: Number of neighbors for local structure
            min_dist: Minimum distance between points
            metric: Distance metric
            random_state: Random seed
            labels: Optional labels for molecules

        Returns:
            ChemicalSpaceEmbedding object
        """
        fps, valid_smiles, valid_indices = self.compute_fingerprints(smiles_list)

        if len(fps) == 0:
            raise ValueError("No valid molecules found")

        # Adjust n_neighbors if needed
        effective_n_neighbors = min(n_neighbors, len(fps) - 1)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        coordinates = reducer.fit_transform(fps)

        valid_labels = None
        if labels is not None:
            valid_labels = labels[valid_indices]

        return ChemicalSpaceEmbedding(
            coordinates=coordinates,
            smiles_list=valid_smiles,
            method="UMAP",
            fingerprint_type=self.fingerprint_type,
            labels=valid_labels,
            metadata={
                "n_neighbors": effective_n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
            },
        )

    def embed(
        self,
        smiles_list: list[str],
        method: str = "tsne",
        n_components: int = 2,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ChemicalSpaceEmbedding:
        """
        Embed molecules using specified method.

        Args:
            smiles_list: List of SMILES strings
            method: Embedding method ('tsne', 'pca', 'umap')
            n_components: Number of dimensions
            labels: Optional labels for molecules
            **kwargs: Additional method-specific arguments

        Returns:
            ChemicalSpaceEmbedding object
        """
        method = method.lower()

        if method == "tsne":
            return self.embed_tsne(smiles_list, n_components, labels=labels, **kwargs)
        elif method == "pca":
            return self.embed_pca(smiles_list, n_components, labels=labels)
        elif method == "umap":
            return self.embed_umap(smiles_list, n_components, labels=labels, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'")


def visualize_chemical_space_plotly(
    embedding: ChemicalSpaceEmbedding,
    color_by: Optional[np.ndarray] = None,
    hover_data: Optional[dict] = None,
    title: str = "Chemical Space",
    colorscale: str = "Viridis",
) -> "plotly.graph_objects.Figure":
    """
    Create interactive plotly visualization of chemical space.

    Args:
        embedding: ChemicalSpaceEmbedding object
        color_by: Optional array to color points by
        hover_data: Optional additional data for hover
        title: Plot title
        colorscale: Plotly colorscale name

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import plotly.express as px

    coords = embedding.coordinates
    is_3d = coords.shape[1] == 3

    # Prepare color
    if color_by is None and embedding.labels is not None:
        color_by = embedding.labels

    # Prepare hover text
    hover_text = [f"SMILES: {s}" for s in embedding.smiles_list]
    if hover_data:
        for key, values in hover_data.items():
            for i, v in enumerate(values):
                hover_text[i] += f"<br>{key}: {v}"

    if is_3d:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color_by,
                        colorscale=colorscale,
                        showscale=color_by is not None,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=color_by,
                        colorscale=colorscale,
                        showscale=color_by is not None,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                )
            ]
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"{embedding.method} 1",
        yaxis_title=f"{embedding.method} 2",
    )

    return fig


def compute_chemical_space_density(
    embedding: ChemicalSpaceEmbedding,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute density of molecules in chemical space.

    Args:
        embedding: ChemicalSpaceEmbedding object
        grid_size: Size of density grid

    Returns:
        Tuple of (x_grid, y_grid, density)
    """
    from scipy.stats import gaussian_kde

    coords = embedding.coordinates[:, :2]  # Use first 2 dimensions

    # Create grid
    x_min, x_max = coords[:, 0].min() - 1, coords[:, 0].max() + 1
    y_min, y_max = coords[:, 1].min() - 1, coords[:, 1].max() + 1

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Compute KDE
    kernel = gaussian_kde(coords.T)
    density = kernel(positions).reshape(grid_size, grid_size)

    return x_grid, y_grid, density
