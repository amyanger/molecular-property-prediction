"""Activity cliff detection utilities for molecular property prediction."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivityCliff:
    """Container for an activity cliff pair."""

    smiles1: str
    smiles2: str
    similarity: float
    activity1: float
    activity2: float
    activity_diff: float
    sali_score: float  # Structure-Activity Landscape Index
    index1: int
    index2: int


@dataclass
class ActivityCliffAnalysis:
    """Container for activity cliff analysis results."""

    total_pairs: int
    num_cliffs: int
    cliff_fraction: float
    mean_sali: float
    max_sali: float
    cliffs: list[ActivityCliff] = field(default_factory=list)


class ActivityCliffDetector:
    """
    Detect activity cliffs in molecular datasets.

    Activity cliffs are pairs of structurally similar molecules
    with significantly different activities.

    Uses the Structure-Activity Landscape Index (SALI):
    SALI = |activity_diff| / (1 - similarity)

    Args:
        similarity_threshold: Minimum Tanimoto similarity for cliff
        activity_threshold: Minimum activity difference for cliff
        fingerprint_radius: Morgan fingerprint radius
        fingerprint_bits: Number of fingerprint bits
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        activity_threshold: float = 1.0,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
    ):
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits

    def _get_fingerprint(self, smiles: str) -> Optional[DataStructs.ExplicitBitVect]:
        """Generate Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fingerprint_radius, nBits=self.fingerprint_bits
        )

    def compute_sali(
        self,
        similarity: float,
        activity_diff: float,
    ) -> float:
        """
        Compute Structure-Activity Landscape Index.

        Args:
            similarity: Tanimoto similarity between molecules
            activity_diff: Absolute difference in activities

        Returns:
            SALI score
        """
        if similarity >= 1.0:
            return 0.0
        return activity_diff / (1.0 - similarity)

    def detect(
        self,
        smiles_list: list[str],
        activities: np.ndarray,
        task_idx: int = 0,
        max_pairs: Optional[int] = None,
    ) -> ActivityCliffAnalysis:
        """
        Detect activity cliffs in a dataset.

        Args:
            smiles_list: List of SMILES strings
            activities: Activity values (N,) or (N, num_tasks)
            task_idx: Task index for multi-task data
            max_pairs: Maximum number of cliffs to return

        Returns:
            ActivityCliffAnalysis object
        """
        # Handle multi-task activities
        if activities.ndim == 2:
            activities = activities[:, task_idx]

        # Generate fingerprints
        fingerprints = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            fp = self._get_fingerprint(smiles)
            if fp is not None and activities[i] >= 0:  # Valid molecule and activity
                fingerprints.append(fp)
                valid_indices.append(i)

        n = len(fingerprints)
        logger.info(f"Computing pairwise similarities for {n} molecules")

        # Find activity cliffs
        cliffs = []
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                idx1, idx2 = valid_indices[i], valid_indices[j]
                total_pairs += 1

                # Compute similarity
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

                if similarity < self.similarity_threshold:
                    continue

                # Check activity difference
                activity_diff = abs(activities[idx1] - activities[idx2])

                if activity_diff < self.activity_threshold:
                    continue

                # Compute SALI
                sali = self.compute_sali(similarity, activity_diff)

                cliffs.append(ActivityCliff(
                    smiles1=smiles_list[idx1],
                    smiles2=smiles_list[idx2],
                    similarity=similarity,
                    activity1=float(activities[idx1]),
                    activity2=float(activities[idx2]),
                    activity_diff=activity_diff,
                    sali_score=sali,
                    index1=idx1,
                    index2=idx2,
                ))

        # Sort by SALI score
        cliffs.sort(key=lambda x: x.sali_score, reverse=True)

        if max_pairs:
            cliffs = cliffs[:max_pairs]

        # Compute statistics
        num_cliffs = len(cliffs)
        cliff_fraction = num_cliffs / total_pairs if total_pairs > 0 else 0.0
        mean_sali = np.mean([c.sali_score for c in cliffs]) if cliffs else 0.0
        max_sali = max(c.sali_score for c in cliffs) if cliffs else 0.0

        return ActivityCliffAnalysis(
            total_pairs=total_pairs,
            num_cliffs=num_cliffs,
            cliff_fraction=cliff_fraction,
            mean_sali=mean_sali,
            max_sali=max_sali,
            cliffs=cliffs,
        )

    def get_cliff_molecules(
        self,
        smiles_list: list[str],
        activities: np.ndarray,
        task_idx: int = 0,
    ) -> Tuple[list[str], list[int]]:
        """
        Get molecules involved in activity cliffs.

        Args:
            smiles_list: List of SMILES strings
            activities: Activity values
            task_idx: Task index

        Returns:
            Tuple of (cliff_smiles, cliff_indices)
        """
        analysis = self.detect(smiles_list, activities, task_idx)

        cliff_indices = set()
        for cliff in analysis.cliffs:
            cliff_indices.add(cliff.index1)
            cliff_indices.add(cliff.index2)

        cliff_indices = sorted(cliff_indices)
        cliff_smiles = [smiles_list[i] for i in cliff_indices]

        return cliff_smiles, cliff_indices


class ActivityCliffSampler:
    """
    Sample training data with awareness of activity cliffs.

    Ensures that cliff pairs are either both in training or both
    in test to avoid information leakage.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        activity_threshold: float = 1.0,
    ):
        self.detector = ActivityCliffDetector(
            similarity_threshold=similarity_threshold,
            activity_threshold=activity_threshold,
        )

    def split(
        self,
        smiles_list: list[str],
        activities: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[list[int], list[int]]:
        """
        Split data while keeping cliff pairs together.

        Args:
            smiles_list: List of SMILES strings
            activities: Activity values
            test_size: Fraction of data for test
            random_state: Random seed

        Returns:
            Tuple of (train_indices, test_indices)
        """
        np.random.seed(random_state)

        # Detect cliffs
        analysis = self.detector.detect(smiles_list, activities)

        # Build graph of cliff pairs
        cliff_graph = {}
        for cliff in analysis.cliffs:
            if cliff.index1 not in cliff_graph:
                cliff_graph[cliff.index1] = set()
            if cliff.index2 not in cliff_graph:
                cliff_graph[cliff.index2] = set()
            cliff_graph[cliff.index1].add(cliff.index2)
            cliff_graph[cliff.index2].add(cliff.index1)

        # Find connected components (molecules that should stay together)
        visited = set()
        components = []

        def dfs(node: int, component: list):
            visited.add(node)
            component.append(node)
            for neighbor in cliff_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for idx in range(len(smiles_list)):
            if idx not in visited:
                component = []
                dfs(idx, component)
                components.append(component)

        # Shuffle components
        np.random.shuffle(components)

        # Assign components to train/test
        train_indices = []
        test_indices = []
        total = len(smiles_list)
        test_count = 0

        for component in components:
            if test_count / total < test_size:
                test_indices.extend(component)
                test_count += len(component)
            else:
                train_indices.extend(component)

        return train_indices, test_indices


def visualize_activity_cliff(cliff: ActivityCliff) -> str:
    """
    Generate SVG visualization of an activity cliff pair.

    Args:
        cliff: ActivityCliff object

    Returns:
        SVG string
    """
    from rdkit.Chem import Draw

    mol1 = Chem.MolFromSmiles(cliff.smiles1)
    mol2 = Chem.MolFromSmiles(cliff.smiles2)

    if mol1 is None or mol2 is None:
        return ""

    # Find common substructure
    from rdkit.Chem import rdFMCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=1)

    if mcs_result.smartsString:
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        match1 = mol1.GetSubstructMatch(mcs_mol)
        match2 = mol2.GetSubstructMatch(mcs_mol)
    else:
        match1 = match2 = []

    # Draw molecules with MCS highlighted
    legends = [
        f"Activity: {cliff.activity1:.2f}",
        f"Activity: {cliff.activity2:.2f}",
    ]

    img = Draw.MolsToGridImage(
        [mol1, mol2],
        molsPerRow=2,
        subImgSize=(300, 300),
        legends=legends,
        highlightAtomLists=[match1, match2],
    )

    return img


def compute_sali_matrix(
    smiles_list: list[str],
    activities: np.ndarray,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
) -> np.ndarray:
    """
    Compute pairwise SALI matrix.

    Args:
        smiles_list: List of SMILES strings
        activities: Activity values
        fingerprint_radius: Morgan fingerprint radius
        fingerprint_bits: Number of fingerprint bits

    Returns:
        SALI matrix (N, N)
    """
    n = len(smiles_list)
    sali_matrix = np.zeros((n, n))

    # Generate fingerprints
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, fingerprint_radius, nBits=fingerprint_bits
            )
        else:
            fp = None
        fingerprints.append(fp)

    # Compute pairwise SALI
    for i in range(n):
        for j in range(i + 1, n):
            if fingerprints[i] is None or fingerprints[j] is None:
                continue

            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            activity_diff = abs(activities[i] - activities[j])

            if similarity < 1.0:
                sali = activity_diff / (1.0 - similarity)
            else:
                sali = 0.0

            sali_matrix[i, j] = sali
            sali_matrix[j, i] = sali

    return sali_matrix
