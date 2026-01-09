"""Active learning utilities for intelligent sample selection."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningQuery:
    """Query result from active learning strategy."""

    indices: np.ndarray
    scores: np.ndarray
    strategy: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ActiveLearningState:
    """State tracking for active learning loop."""

    labeled_indices: list
    unlabeled_indices: list
    query_history: list
    iteration: int = 0
    total_queries: int = 0


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def score(
        self,
        predictions: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute acquisition scores for unlabeled samples.

        Args:
            predictions: Model predictions for unlabeled samples

        Returns:
            Acquisition scores (higher = more informative)
        """
        pass


class UncertaintySampling(AcquisitionFunction):
    """
    Uncertainty-based sampling strategies.

    Selects samples where the model is most uncertain.

    Args:
        method: Uncertainty method ('entropy', 'margin', 'least_confident')
    """

    def __init__(self, method: str = "entropy"):
        self.method = method

    def score(
        self,
        predictions: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute uncertainty scores."""
        # Clip predictions to avoid numerical issues
        probs = np.clip(predictions, 1e-7, 1 - 1e-7)

        if self.method == "entropy":
            # Shannon entropy
            entropy = -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
            if len(entropy.shape) > 1:
                return entropy.mean(axis=1)  # Average across tasks
            return entropy

        elif self.method == "margin":
            # Margin sampling (distance to decision boundary)
            margin = np.abs(probs - 0.5)
            if len(margin.shape) > 1:
                return 1 - margin.mean(axis=1)  # Convert to uncertainty
            return 1 - margin

        elif self.method == "least_confident":
            # Confidence in predicted class
            confidence = np.maximum(probs, 1 - probs)
            if len(confidence.shape) > 1:
                return 1 - confidence.mean(axis=1)
            return 1 - confidence

        else:
            raise ValueError(f"Unknown method: {self.method}")


class DiversitySampling(AcquisitionFunction):
    """
    Diversity-based sampling using feature space clustering.

    Selects samples that are most diverse in feature space.

    Args:
        n_clusters: Number of clusters for K-means
    """

    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def score(
        self,
        predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute diversity scores based on clustering.

        Args:
            predictions: Model predictions
            features: Feature representations for samples

        Returns:
            Diversity scores
        """
        if features is None:
            features = predictions

        # Cluster the feature space
        n_clusters = min(self.n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Score based on distance to cluster center
        distances = kmeans.transform(features)
        min_distances = distances.min(axis=1)

        return min_distances


class BALDSampling(AcquisitionFunction):
    """
    Bayesian Active Learning by Disagreement (BALD).

    Uses Monte Carlo Dropout to estimate model uncertainty.
    Selects samples with highest mutual information.

    Reference: Gal et al., "Deep Bayesian Active Learning with Image Data", ICML 2017
    """

    def score(
        self,
        predictions: np.ndarray,
        mc_predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute BALD scores from MC Dropout predictions.

        Args:
            predictions: Mean predictions [N, T] where T is num tasks
            mc_predictions: MC samples [N, M, T] where M is num MC samples

        Returns:
            BALD scores
        """
        if mc_predictions is None:
            # Fall back to entropy if no MC predictions
            probs = np.clip(predictions, 1e-7, 1 - 1e-7)
            entropy = -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
            if len(entropy.shape) > 1:
                return entropy.mean(axis=1)
            return entropy

        # Compute mutual information (BALD score)
        # I(y; w | x) = H(y | x) - E_w[H(y | x, w)]

        # Mean prediction entropy
        mean_preds = mc_predictions.mean(axis=1)
        mean_preds = np.clip(mean_preds, 1e-7, 1 - 1e-7)
        predictive_entropy = -(
            mean_preds * np.log2(mean_preds) +
            (1 - mean_preds) * np.log2(1 - mean_preds)
        )

        # Expected entropy
        mc_preds = np.clip(mc_predictions, 1e-7, 1 - 1e-7)
        sample_entropies = -(
            mc_preds * np.log2(mc_preds) +
            (1 - mc_preds) * np.log2(1 - mc_preds)
        )
        expected_entropy = sample_entropies.mean(axis=1)

        # Mutual information
        bald_scores = predictive_entropy - expected_entropy

        if len(bald_scores.shape) > 1:
            return bald_scores.mean(axis=1)
        return bald_scores


class CoreSetSampling(AcquisitionFunction):
    """
    Core-set selection for active learning.

    Selects samples that provide maximum coverage of the feature space.

    Reference: Sener & Savarese, "Active Learning for CNNs: A Core-Set Approach", ICLR 2018
    """

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric

    def score(
        self,
        predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        labeled_features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute core-set scores (max distance to labeled set).

        Args:
            predictions: Model predictions (unused, for interface compatibility)
            features: Features of unlabeled samples
            labeled_features: Features of already labeled samples

        Returns:
            Core-set scores
        """
        if features is None:
            features = predictions

        if labeled_features is None or len(labeled_features) == 0:
            # No labeled samples yet - return uniform
            return np.ones(len(features))

        # Compute distance to nearest labeled sample
        distances = pairwise_distances(
            features, labeled_features, metric=self.metric
        )
        min_distances = distances.min(axis=1)

        return min_distances


class BatchBALD(AcquisitionFunction):
    """
    BatchBALD for batch-mode active learning.

    Jointly selects a batch of samples that maximize mutual information.

    Reference: Kirsch et al., "BatchBALD: Efficient and Diverse Batch Acquisition", NeurIPS 2019
    """

    def __init__(self, batch_size: int = 10, num_samples: int = 100):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def score(
        self,
        predictions: np.ndarray,
        mc_predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute approximate BatchBALD scores.

        Uses greedy approximation for efficiency.
        """
        if mc_predictions is None:
            # Fall back to regular uncertainty
            probs = np.clip(predictions, 1e-7, 1 - 1e-7)
            entropy = -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
            if len(entropy.shape) > 1:
                return entropy.mean(axis=1)
            return entropy

        # Greedy batch selection based on BALD scores with diversity
        bald_sampler = BALDSampling()
        bald_scores = bald_sampler.score(predictions, mc_predictions)

        # Add diversity bonus using prediction variance
        pred_std = mc_predictions.std(axis=1)
        if len(pred_std.shape) > 1:
            diversity = pred_std.mean(axis=1)
        else:
            diversity = pred_std

        # Combined score
        return bald_scores + 0.1 * diversity


class ExpectedModelChange(AcquisitionFunction):
    """
    Expected Model Change (EMC) acquisition function.

    Selects samples that would cause the largest change in model parameters.

    Note: Requires gradient computation capability.
    """

    def score(
        self,
        predictions: np.ndarray,
        gradients: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute expected model change scores.

        Args:
            predictions: Model predictions
            gradients: Gradient norms for each sample

        Returns:
            EMC scores
        """
        if gradients is None:
            # Fall back to uncertainty if no gradients
            probs = np.clip(predictions, 1e-7, 1 - 1e-7)
            return np.abs(probs - 0.5).mean(axis=1) if len(probs.shape) > 1 else np.abs(probs - 0.5)

        # Gradient magnitude as proxy for expected model change
        if len(gradients.shape) > 1:
            return np.linalg.norm(gradients, axis=1)
        return gradients


class QueryByCommittee(AcquisitionFunction):
    """
    Query by Committee (QBC) sampling.

    Uses disagreement among ensemble members to select samples.

    Args:
        measure: Disagreement measure ('vote_entropy', 'kl_divergence')
    """

    def __init__(self, measure: str = "vote_entropy"):
        self.measure = measure

    def score(
        self,
        predictions: np.ndarray,
        committee_predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute committee disagreement scores.

        Args:
            predictions: Mean predictions
            committee_predictions: Predictions from each committee member [N, C, T]
                                  where C is number of committee members

        Returns:
            Disagreement scores
        """
        if committee_predictions is None:
            # No committee - use uncertainty
            probs = np.clip(predictions, 1e-7, 1 - 1e-7)
            entropy = -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
            if len(entropy.shape) > 1:
                return entropy.mean(axis=1)
            return entropy

        if self.measure == "vote_entropy":
            # Binary vote entropy
            votes = (committee_predictions > 0.5).astype(float)
            vote_ratios = votes.mean(axis=1)
            vote_ratios = np.clip(vote_ratios, 1e-7, 1 - 1e-7)
            entropy = -(
                vote_ratios * np.log2(vote_ratios) +
                (1 - vote_ratios) * np.log2(1 - vote_ratios)
            )
            if len(entropy.shape) > 1:
                return entropy.mean(axis=1)
            return entropy

        elif self.measure == "kl_divergence":
            # Average KL divergence between members and consensus
            mean_preds = committee_predictions.mean(axis=1)
            mean_preds = np.clip(mean_preds, 1e-7, 1 - 1e-7)

            kl_divs = []
            for c in range(committee_predictions.shape[1]):
                member_preds = np.clip(committee_predictions[:, c], 1e-7, 1 - 1e-7)
                kl = (
                    member_preds * np.log(member_preds / mean_preds) +
                    (1 - member_preds) * np.log((1 - member_preds) / (1 - mean_preds))
                )
                kl_divs.append(kl)

            kl_divs = np.stack(kl_divs, axis=1)
            avg_kl = kl_divs.mean(axis=1)

            if len(avg_kl.shape) > 1:
                return avg_kl.mean(axis=1)
            return avg_kl

        else:
            raise ValueError(f"Unknown measure: {self.measure}")


class ActiveLearner:
    """
    Active learning loop manager.

    Manages the active learning process including querying,
    labeling simulation, and model retraining.

    Args:
        model: Model to train
        acquisition_fn: Acquisition function for sample selection
        initial_labeled: Initial labeled samples indices
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        acquisition_fn: AcquisitionFunction,
        initial_labeled: Optional[list[int]] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.acquisition_fn = acquisition_fn
        self.device = device

        self.state = ActiveLearningState(
            labeled_indices=list(initial_labeled) if initial_labeled else [],
            unlabeled_indices=[],
            query_history=[],
        )

    def initialize(
        self,
        dataset_size: int,
        initial_size: int = 100,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize active learning state with random initial samples.

        Args:
            dataset_size: Total number of samples
            initial_size: Number of initial labeled samples
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)

        if not self.state.labeled_indices:
            self.state.labeled_indices = list(
                np.random.choice(dataset_size, initial_size, replace=False)
            )

        all_indices = set(range(dataset_size))
        self.state.unlabeled_indices = list(
            all_indices - set(self.state.labeled_indices)
        )

        logger.info(f"Initialized with {len(self.state.labeled_indices)} labeled, "
                   f"{len(self.state.unlabeled_indices)} unlabeled samples")

    def query(
        self,
        n_samples: int,
        features: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ActiveLearningQuery:
        """
        Query for next samples to label.

        Args:
            n_samples: Number of samples to query
            features: Feature representations (for diversity-based methods)
            predictions: Model predictions on unlabeled set
            **kwargs: Additional arguments for acquisition function

        Returns:
            ActiveLearningQuery with selected indices
        """
        if predictions is None:
            raise ValueError("Predictions required for querying")

        # Get unlabeled predictions
        unlabeled_indices = np.array(self.state.unlabeled_indices)

        # Compute acquisition scores
        labeled_features = None
        if features is not None and self.state.labeled_indices:
            labeled_features = features[self.state.labeled_indices]
            unlabeled_features = features[unlabeled_indices]
            kwargs["features"] = unlabeled_features
            kwargs["labeled_features"] = labeled_features

        scores = self.acquisition_fn.score(predictions, **kwargs)

        # Select top-k samples
        n_samples = min(n_samples, len(unlabeled_indices))
        top_indices = np.argsort(scores)[-n_samples:][::-1]

        selected_indices = unlabeled_indices[top_indices]
        selected_scores = scores[top_indices]

        query = ActiveLearningQuery(
            indices=selected_indices,
            scores=selected_scores,
            strategy=self.acquisition_fn.__class__.__name__,
        )

        # Update state
        self.state.query_history.append(query)
        self.state.iteration += 1
        self.state.total_queries += n_samples

        return query

    def update(self, new_labeled_indices: Union[list, np.ndarray]) -> None:
        """
        Update state with newly labeled samples.

        Args:
            new_labeled_indices: Indices of newly labeled samples
        """
        new_labeled = set(new_labeled_indices)
        self.state.labeled_indices.extend(new_labeled_indices)
        self.state.unlabeled_indices = [
            i for i in self.state.unlabeled_indices if i not in new_labeled
        ]

        logger.info(f"Updated: {len(self.state.labeled_indices)} labeled, "
                   f"{len(self.state.unlabeled_indices)} unlabeled")

    def get_train_indices(self) -> list:
        """Get indices of labeled samples for training."""
        return self.state.labeled_indices.copy()


def get_mc_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 10,
    device: str = "cuda",
) -> np.ndarray:
    """
    Get Monte Carlo Dropout predictions.

    Args:
        model: Model with dropout layers
        dataloader: Data loader for predictions
        n_samples: Number of MC samples
        device: Device for computation

    Returns:
        MC predictions array [N, M, T] where M is n_samples
    """
    model.to(device)
    model.train()  # Enable dropout

    all_predictions = []

    for _ in range(n_samples):
        sample_preds = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                    else:
                        inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
                    logits = model(inputs) if isinstance(inputs, torch.Tensor) else model(*inputs)
                else:
                    logits = model(batch.to(device))

                probs = torch.sigmoid(logits).cpu().numpy()
                sample_preds.append(probs)

        all_predictions.append(np.concatenate(sample_preds, axis=0))

    # Stack: [M, N, T] -> transpose to [N, M, T]
    mc_predictions = np.stack(all_predictions, axis=1)

    model.eval()
    return mc_predictions


def active_learning_loop(
    model: nn.Module,
    train_fn: Callable,
    predict_fn: Callable,
    dataset: torch.utils.data.Dataset,
    acquisition: str = "uncertainty",
    initial_size: int = 100,
    query_size: int = 50,
    max_iterations: int = 20,
    device: str = "cuda",
) -> dict:
    """
    Run full active learning loop.

    Args:
        model: Model to train
        train_fn: Training function (takes indices, returns metrics)
        predict_fn: Prediction function (takes indices, returns predictions)
        dataset: Full dataset
        acquisition: Acquisition strategy name
        initial_size: Initial labeled pool size
        query_size: Samples to query per iteration
        max_iterations: Maximum AL iterations
        device: Device for computation

    Returns:
        Dict with learning curves and final metrics
    """
    # Set up acquisition function
    acquisition_fns = {
        "uncertainty": UncertaintySampling("entropy"),
        "margin": UncertaintySampling("margin"),
        "least_confident": UncertaintySampling("least_confident"),
        "diversity": DiversitySampling(),
        "bald": BALDSampling(),
        "coreset": CoreSetSampling(),
        "qbc": QueryByCommittee(),
    }

    if acquisition not in acquisition_fns:
        raise ValueError(f"Unknown acquisition: {acquisition}")

    acq_fn = acquisition_fns[acquisition]

    # Initialize active learner
    learner = ActiveLearner(model, acq_fn, device=device)
    learner.initialize(len(dataset), initial_size)

    # Learning curves
    history = {
        "iterations": [],
        "labeled_count": [],
        "train_metrics": [],
        "query_scores": [],
    }

    for iteration in range(max_iterations):
        logger.info(f"\n=== Active Learning Iteration {iteration + 1}/{max_iterations} ===")

        # Train on current labeled set
        train_indices = learner.get_train_indices()
        train_metrics = train_fn(train_indices)

        logger.info(f"Trained on {len(train_indices)} samples")

        # Get predictions on unlabeled set
        unlabeled_indices = learner.state.unlabeled_indices
        if not unlabeled_indices:
            logger.info("No more unlabeled samples")
            break

        predictions = predict_fn(unlabeled_indices)

        # Query new samples
        query = learner.query(query_size, predictions=predictions)

        logger.info(f"Queried {len(query.indices)} samples with {acquisition} strategy")
        logger.info(f"Top acquisition scores: {query.scores[:5]}")

        # Update with queried samples
        learner.update(query.indices)

        # Record history
        history["iterations"].append(iteration + 1)
        history["labeled_count"].append(len(learner.get_train_indices()))
        history["train_metrics"].append(train_metrics)
        history["query_scores"].append(query.scores.mean())

    return history
