"""Molecular similarity search and comparison utilities."""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit import SimDivFilters
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Container for similarity search results."""

    smiles: str
    similarity: float
    index: int
    rank: int


@dataclass
class MolecularSimilarityMatrix:
    """Container for pairwise similarity matrix."""

    matrix: np.ndarray
    smiles_list: list[str]
    fingerprint_type: str


class MolecularSimilarity:
    """
    Compute molecular similarity using various fingerprint methods.

    Supports multiple fingerprint types and similarity metrics.

    Args:
        fingerprint_type: Type of fingerprint to use
        radius: Radius for circular fingerprints (Morgan/ECFP)
        n_bits: Number of bits for fingerprint
    """

    FINGERPRINT_TYPES = [
        "morgan",
        "rdkit",
        "maccs",
        "atom_pair",
        "topological_torsion",
        "avalon",
    ]

    SIMILARITY_METRICS = [
        "tanimoto",
        "dice",
        "cosine",
        "sokal",
        "russel",
        "kulczynski",
        "mcconnaughey",
    ]

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
    ):
        if fingerprint_type not in self.FINGERPRINT_TYPES:
            raise ValueError(
                f"Unknown fingerprint type: {fingerprint_type}. "
                f"Available: {self.FINGERPRINT_TYPES}"
            )

        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits

    def get_fingerprint(self, smiles: str) -> Optional[DataStructs.ExplicitBitVect]:
        """
        Generate fingerprint for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            RDKit fingerprint or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.fingerprint_type == "morgan":
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
        elif self.fingerprint_type == "rdkit":
            return Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        elif self.fingerprint_type == "maccs":
            return AllChem.GetMACCSKeysFingerprint(mol)
        elif self.fingerprint_type == "atom_pair":
            return Pairs.GetAtomPairFingerprintAsBitVect(mol)
        elif self.fingerprint_type == "topological_torsion":
            return Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        elif self.fingerprint_type == "avalon":
            try:
                from rdkit.Avalon import pyAvalonTools
                return pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits)
            except ImportError:
                logger.warning("Avalon fingerprints not available, using Morgan")
                return AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

    def compute_similarity(
        self,
        smiles1: str,
        smiles2: str,
        metric: str = "tanimoto",
    ) -> float:
        """
        Compute similarity between two molecules.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            metric: Similarity metric to use

        Returns:
            Similarity score (0 to 1)
        """
        fp1 = self.get_fingerprint(smiles1)
        fp2 = self.get_fingerprint(smiles2)

        if fp1 is None or fp2 is None:
            return 0.0

        return self._compute_fingerprint_similarity(fp1, fp2, metric)

    def _compute_fingerprint_similarity(
        self,
        fp1: DataStructs.ExplicitBitVect,
        fp2: DataStructs.ExplicitBitVect,
        metric: str = "tanimoto",
    ) -> float:
        """Compute similarity between two fingerprints."""
        if metric == "tanimoto":
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif metric == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)
        elif metric == "cosine":
            return DataStructs.CosineSimilarity(fp1, fp2)
        elif metric == "sokal":
            return DataStructs.SokalSimilarity(fp1, fp2)
        elif metric == "russel":
            return DataStructs.RusselSimilarity(fp1, fp2)
        elif metric == "kulczynski":
            return DataStructs.KulczynskiSimilarity(fp1, fp2)
        elif metric == "mcconnaughey":
            return DataStructs.McConnaugheySimilarity(fp1, fp2)
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Available: {self.SIMILARITY_METRICS}"
            )

    def find_similar(
        self,
        query_smiles: str,
        database_smiles: list[str],
        metric: str = "tanimoto",
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[SimilarityResult]:
        """
        Find similar molecules in a database.

        Args:
            query_smiles: Query molecule SMILES
            database_smiles: List of database SMILES
            metric: Similarity metric
            top_k: Number of top similar molecules to return
            threshold: Minimum similarity threshold

        Returns:
            List of SimilarityResult sorted by similarity (descending)
        """
        query_fp = self.get_fingerprint(query_smiles)
        if query_fp is None:
            logger.warning(f"Invalid query SMILES: {query_smiles}")
            return []

        # Compute similarities
        similarities = []
        for i, smiles in enumerate(database_smiles):
            fp = self.get_fingerprint(smiles)
            if fp is not None:
                sim = self._compute_fingerprint_similarity(query_fp, fp, metric)
                if sim >= threshold:
                    similarities.append((smiles, sim, i))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for rank, (smiles, sim, idx) in enumerate(similarities[:top_k]):
            results.append(
                SimilarityResult(
                    smiles=smiles,
                    similarity=sim,
                    index=idx,
                    rank=rank + 1,
                )
            )

        return results

    def compute_similarity_matrix(
        self,
        smiles_list: list[str],
        metric: str = "tanimoto",
    ) -> MolecularSimilarityMatrix:
        """
        Compute pairwise similarity matrix for a set of molecules.

        Args:
            smiles_list: List of SMILES strings
            metric: Similarity metric

        Returns:
            MolecularSimilarityMatrix object
        """
        n = len(smiles_list)
        matrix = np.zeros((n, n))

        # Precompute fingerprints
        fingerprints = []
        for smiles in smiles_list:
            fingerprints.append(self.get_fingerprint(smiles))

        # Compute pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if fingerprints[i] is None or fingerprints[j] is None:
                    sim = 0.0
                elif i == j:
                    sim = 1.0
                else:
                    sim = self._compute_fingerprint_similarity(
                        fingerprints[i], fingerprints[j], metric
                    )
                matrix[i, j] = sim
                matrix[j, i] = sim

        return MolecularSimilarityMatrix(
            matrix=matrix,
            smiles_list=smiles_list,
            fingerprint_type=self.fingerprint_type,
        )


class DiversityPicker:
    """
    Select diverse subsets of molecules using various algorithms.

    Args:
        fingerprint_type: Type of fingerprint to use
        radius: Radius for circular fingerprints
        n_bits: Number of bits for fingerprint
    """

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.similarity = MolecularSimilarity(fingerprint_type, radius, n_bits)

    def maxmin_picker(
        self,
        smiles_list: list[str],
        n_picks: int,
        seed: Optional[str] = None,
    ) -> list[int]:
        """
        Select diverse molecules using MaxMin algorithm.

        Iteratively selects molecules that maximize the minimum
        distance to already selected molecules.

        Args:
            smiles_list: List of SMILES strings
            n_picks: Number of molecules to select
            seed: Optional seed molecule SMILES

        Returns:
            Indices of selected molecules
        """
        # Get fingerprints
        fps = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            fp = self.similarity.get_fingerprint(smiles)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(i)

        if len(fps) < n_picks:
            logger.warning(f"Only {len(fps)} valid molecules, returning all")
            return valid_indices

        # Use RDKit's MaxMinPicker
        picker = SimDivFilters.MaxMinPicker()

        # If seed provided, find its index
        first_pick = -1
        if seed:
            seed_fp = self.similarity.get_fingerprint(seed)
            if seed_fp is not None:
                # Find most similar molecule as starting point
                max_sim = -1
                for i, fp in enumerate(fps):
                    sim = DataStructs.TanimotoSimilarity(seed_fp, fp)
                    if sim > max_sim:
                        max_sim = sim
                        first_pick = i

        # Pick diverse molecules
        picks = picker.LazyBitVectorPick(
            fps, len(fps), n_picks, firstPicks=[first_pick] if first_pick >= 0 else []
        )

        return [valid_indices[i] for i in picks]

    def cluster_picker(
        self,
        smiles_list: list[str],
        n_picks: int,
        n_clusters: Optional[int] = None,
    ) -> list[int]:
        """
        Select diverse molecules using clustering.

        Clusters molecules and picks centroids from each cluster.

        Args:
            smiles_list: List of SMILES strings
            n_picks: Number of molecules to select
            n_clusters: Number of clusters (defaults to n_picks)

        Returns:
            Indices of selected molecules
        """
        from sklearn.cluster import KMeans

        if n_clusters is None:
            n_clusters = n_picks

        # Compute fingerprints as numpy arrays
        fps_arrays = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            fp = self.similarity.get_fingerprint(smiles)
            if fp is not None:
                arr = np.zeros(fp.GetNumBits())
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps_arrays.append(arr)
                valid_indices.append(i)

        if len(fps_arrays) < n_picks:
            return valid_indices

        fps_matrix = np.array(fps_arrays)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fps_matrix)

        # Pick molecule closest to each centroid
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(fps_matrix[cluster_mask] - centroid, axis=1)
            best_in_cluster = cluster_indices[np.argmin(distances)]
            selected.append(valid_indices[best_in_cluster])

            if len(selected) >= n_picks:
                break

        return selected[:n_picks]


def compute_tanimoto_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto distance between two fingerprint arrays."""
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    if union == 0:
        return 1.0
    return 1.0 - intersection / union


def fingerprint_to_numpy(fp: DataStructs.ExplicitBitVect) -> np.ndarray:
    """Convert RDKit fingerprint to numpy array."""
    arr = np.zeros(fp.GetNumBits())
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_tanimoto_similarity(
    query_fp: np.ndarray,
    database_fps: np.ndarray,
) -> np.ndarray:
    """
    Compute Tanimoto similarity between query and database fingerprints.

    Vectorized implementation for efficiency.

    Args:
        query_fp: Query fingerprint (n_bits,)
        database_fps: Database fingerprints (n_molecules, n_bits)

    Returns:
        Similarities (n_molecules,)
    """
    # Intersection: element-wise AND
    intersection = np.logical_and(query_fp, database_fps).sum(axis=1)

    # Union: a + b - intersection
    query_bits = query_fp.sum()
    database_bits = database_fps.sum(axis=1)
    union = query_bits + database_bits - intersection

    # Tanimoto
    similarities = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )

    return similarities
