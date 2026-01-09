"""Curriculum learning utilities for training molecular models."""

import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    strategy: str = "linear"  # "linear", "exponential", "self_paced", "competence"
    start_fraction: float = 0.3
    end_fraction: float = 1.0
    warmup_epochs: int = 5
    pacing_function: str = "linear"  # "linear", "sqrt", "log"


@dataclass
class SampleDifficulty:
    """Difficulty scores for samples."""

    indices: np.ndarray
    scores: np.ndarray
    method: str
    metadata: dict = field(default_factory=dict)


class DifficultyScorer(ABC):
    """Base class for sample difficulty scoring."""

    @abstractmethod
    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Compute difficulty scores for all samples.

        Args:
            model: Model to use for scoring
            dataloader: Data loader with all samples
            device: Device for computation

        Returns:
            Difficulty scores (higher = harder)
        """
        pass


class LossBasedDifficulty(DifficultyScorer):
    """
    Score difficulty based on loss values.

    Samples with higher loss are considered more difficult.
    """

    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> np.ndarray:
        model.to(device)
        model.eval()

        all_losses = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, labels = batch
                    mask = None
                else:
                    inputs, labels, mask = batch[0], batch[1], batch[2] if len(batch) > 2 else None

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                else:
                    inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
                labels = labels.to(device)

                if isinstance(inputs, torch.Tensor):
                    outputs = model(inputs)
                else:
                    outputs = model(*inputs)

                # Per-sample loss
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels.float(), reduction="none"
                )

                if mask is not None:
                    mask = mask.to(device)
                    loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                else:
                    loss = loss.mean(dim=-1)

                all_losses.append(loss.cpu().numpy())

        return np.concatenate(all_losses)


class ConfidenceBasedDifficulty(DifficultyScorer):
    """
    Score difficulty based on prediction confidence.

    Samples with low confidence (close to 0.5) are harder.
    """

    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> np.ndarray:
        model.to(device)
        model.eval()

        all_uncertainties = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, _ = batch
                else:
                    inputs = batch[0]

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                else:
                    inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)

                if isinstance(inputs, torch.Tensor):
                    outputs = model(inputs)
                else:
                    outputs = model(*inputs)

                probs = torch.sigmoid(outputs)

                # Uncertainty = distance from 0.5
                uncertainty = 0.5 - torch.abs(probs - 0.5)
                uncertainty = uncertainty.mean(dim=-1)

                all_uncertainties.append(uncertainty.cpu().numpy())

        return np.concatenate(all_uncertainties)


class GradientNormDifficulty(DifficultyScorer):
    """
    Score difficulty based on gradient norm.

    Samples with larger gradients indicate more learning signal.
    """

    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> np.ndarray:
        model.to(device)
        model.eval()

        all_grad_norms = []

        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels = batch[0], batch[1]

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device).requires_grad_(True)
            labels = labels.to(device)

            if isinstance(inputs, torch.Tensor):
                outputs = model(inputs)
            else:
                inputs_list = list(inputs)
                inputs_list[0] = inputs_list[0].to(device).requires_grad_(True)
                outputs = model(*inputs_list)
                inputs = inputs_list[0]

            # Compute per-sample gradients via loss
            batch_size = labels.size(0)
            grad_norms = []

            for i in range(batch_size):
                model.zero_grad()
                if inputs.grad is not None:
                    inputs.grad.zero_()

                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs[i:i+1], labels[i:i+1].float()
                )
                loss.backward(retain_graph=True)

                if inputs.grad is not None:
                    grad_norm = inputs.grad[i].norm().item()
                else:
                    grad_norm = 0.0
                grad_norms.append(grad_norm)

            all_grad_norms.extend(grad_norms)

        return np.array(all_grad_norms)


class MolecularComplexityDifficulty(DifficultyScorer):
    """
    Score difficulty based on molecular complexity.

    Uses molecular properties like size, rings, and functional groups.
    """

    def __init__(
        self,
        weight_atoms: float = 0.3,
        weight_bonds: float = 0.2,
        weight_rings: float = 0.3,
        weight_heteroatoms: float = 0.2,
    ):
        self.weight_atoms = weight_atoms
        self.weight_bonds = weight_bonds
        self.weight_rings = weight_rings
        self.weight_heteroatoms = weight_heteroatoms

    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> np.ndarray:
        """Score based on graph complexity."""
        all_scores = []

        for batch in dataloader:
            # Assume PyG batch format
            if hasattr(batch, 'x'):
                x = batch.x
                edge_index = batch.edge_index
                batch_idx = batch.batch

                for b in batch_idx.unique():
                    mask = batch_idx == b
                    num_atoms = mask.sum().item()
                    num_edges = ((edge_index[0][None, :] == mask.nonzero()[:, 0][:, None]).any(0)).sum().item()

                    # Estimate complexity
                    complexity = (
                        self.weight_atoms * np.log1p(num_atoms) +
                        self.weight_bonds * np.log1p(num_edges / 2) +
                        self.weight_rings * max(0, (num_edges / 2 - num_atoms + 1)) +
                        self.weight_heteroatoms * 0  # Would need atom types
                    )
                    all_scores.append(complexity)
            else:
                # Fingerprint-based: use bit density
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(inputs, torch.Tensor):
                    densities = inputs.float().mean(dim=-1).cpu().numpy()
                    all_scores.extend(densities)

        return np.array(all_scores)


class CurriculumScheduler:
    """
    Scheduler for curriculum learning.

    Controls the pacing of training data difficulty.

    Args:
        config: Curriculum configuration
        total_epochs: Total training epochs
        num_samples: Total number of samples
    """

    def __init__(
        self,
        config: CurriculumConfig,
        total_epochs: int,
        num_samples: int,
    ):
        self.config = config
        self.total_epochs = total_epochs
        self.num_samples = num_samples
        self.current_epoch = 0

    def get_fraction(self, epoch: Optional[int] = None) -> float:
        """
        Get the fraction of data to use at given epoch.

        Args:
            epoch: Epoch number (uses current if None)

        Returns:
            Fraction of data to use
        """
        if epoch is None:
            epoch = self.current_epoch

        # Warmup phase
        if epoch < self.config.warmup_epochs:
            return self.config.start_fraction

        # Compute progress after warmup
        progress = (epoch - self.config.warmup_epochs) / max(
            1, self.total_epochs - self.config.warmup_epochs
        )
        progress = min(1.0, progress)

        # Pacing function
        if self.config.pacing_function == "linear":
            paced_progress = progress
        elif self.config.pacing_function == "sqrt":
            paced_progress = np.sqrt(progress)
        elif self.config.pacing_function == "log":
            paced_progress = np.log1p(progress * np.e - progress) if progress > 0 else 0
        else:
            paced_progress = progress

        # Interpolate fraction
        fraction = (
            self.config.start_fraction +
            (self.config.end_fraction - self.config.start_fraction) * paced_progress
        )

        return min(self.config.end_fraction, fraction)

    def get_sample_indices(
        self,
        difficulty_scores: np.ndarray,
        epoch: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get indices of samples to use at given epoch.

        Args:
            difficulty_scores: Difficulty scores for all samples
            epoch: Epoch number

        Returns:
            Indices of samples to use
        """
        fraction = self.get_fraction(epoch)
        num_samples = int(self.num_samples * fraction)

        # Sort by difficulty and take easiest samples
        sorted_indices = np.argsort(difficulty_scores)
        return sorted_indices[:num_samples]

    def step(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1


class SelfPacedLearning:
    """
    Self-paced learning with adaptive sample weighting.

    Automatically adjusts sample weights based on current loss.

    Reference: Kumar et al., "Self-Paced Learning with Diversity"

    Args:
        lambda_init: Initial pace parameter
        lambda_growth: Growth rate per epoch
        diversity_weight: Weight for diversity term
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        lambda_growth: float = 1.2,
        diversity_weight: float = 0.1,
    ):
        self.lambda_param = lambda_init
        self.lambda_growth = lambda_growth
        self.diversity_weight = diversity_weight

    def compute_weights(
        self,
        losses: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sample weights based on losses.

        Args:
            losses: Per-sample losses
            features: Optional features for diversity

        Returns:
            Sample weights
        """
        # Hard threshold: v = 1 if loss < lambda, else 0
        # Soft version: v = max(0, 1 - loss/lambda)
        weights = torch.clamp(1 - losses / self.lambda_param, min=0, max=1)

        # Add diversity term if features provided
        if features is not None and self.diversity_weight > 0:
            # Compute pairwise distances
            dist = torch.cdist(features, features)
            # Diversity bonus for dissimilar samples
            diversity = dist.mean(dim=1)
            diversity = diversity / diversity.max()
            weights = weights + self.diversity_weight * diversity

        return weights / weights.sum() * len(weights)

    def step(self) -> None:
        """Increase pace parameter."""
        self.lambda_param *= self.lambda_growth


class CompetenceBasedCurriculum:
    """
    Competence-based curriculum learning.

    Adjusts difficulty based on model competence (performance).

    Reference: Platanios et al., "Competence-based Curriculum Learning"

    Args:
        competence_fn: Function to compute model competence
        difficulty_fn: Function to map competence to difficulty threshold
    """

    def __init__(
        self,
        competence_fn: Optional[Callable] = None,
        difficulty_fn: Optional[Callable] = None,
    ):
        self.competence_fn = competence_fn or self._default_competence
        self.difficulty_fn = difficulty_fn or self._default_difficulty
        self.competence_history = []

    def _default_competence(self, metrics: dict) -> float:
        """Default competence: use AUC or accuracy."""
        if "auc" in metrics:
            return metrics["auc"]
        elif "accuracy" in metrics:
            return metrics["accuracy"]
        return 0.5

    def _default_difficulty(self, competence: float) -> float:
        """Default difficulty threshold based on competence."""
        # Higher competence = can handle harder samples
        return competence ** 2

    def update_competence(self, metrics: dict) -> float:
        """Update competence based on validation metrics."""
        competence = self.competence_fn(metrics)
        self.competence_history.append(competence)
        return competence

    def get_difficulty_threshold(self) -> float:
        """Get current difficulty threshold."""
        if not self.competence_history:
            return 0.0
        competence = np.mean(self.competence_history[-5:])  # Smoothed
        return self.difficulty_fn(competence)

    def filter_samples(
        self,
        difficulty_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Get indices of samples within difficulty threshold.

        Args:
            difficulty_scores: Normalized difficulty scores [0, 1]

        Returns:
            Indices of samples to use
        """
        threshold = self.get_difficulty_threshold()
        return np.where(difficulty_scores <= threshold)[0]


class CurriculumTrainer:
    """
    Trainer with curriculum learning support.

    Args:
        model: Model to train
        difficulty_scorer: Scorer for sample difficulty
        scheduler: Curriculum scheduler
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        difficulty_scorer: DifficultyScorer,
        scheduler: CurriculumScheduler,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.difficulty_scorer = difficulty_scorer
        self.scheduler = scheduler
        self.device = device
        self.difficulty_scores = None

    def compute_difficulty(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Compute and cache difficulty scores."""
        self.difficulty_scores = self.difficulty_scorer.score(
            self.model, dataloader, self.device
        )
        # Normalize to [0, 1]
        self.difficulty_scores = (
            self.difficulty_scores - self.difficulty_scores.min()
        ) / (self.difficulty_scores.max() - self.difficulty_scores.min() + 1e-8)
        return self.difficulty_scores

    def get_curriculum_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        epoch: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Get data loader with curriculum sampling.

        Args:
            dataset: Full dataset
            batch_size: Batch size
            epoch: Current epoch

        Returns:
            DataLoader with curriculum samples
        """
        if self.difficulty_scores is None:
            raise ValueError("Call compute_difficulty first")

        indices = self.scheduler.get_sample_indices(
            self.difficulty_scores, epoch
        )

        subset = torch.utils.data.Subset(dataset, indices)

        return torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
        )

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> dict:
        """
        Train for one epoch with curriculum.

        Args:
            dataloader: Data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels = batch[0], batch[1]

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            else:
                inputs = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            if isinstance(inputs, torch.Tensor):
                outputs = self.model(inputs)
            else:
                outputs = self.model(*inputs)

            loss = criterion(outputs, labels.float())
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / num_batches,
            "fraction": self.scheduler.get_fraction(),
            "epoch": self.scheduler.current_epoch,
        }


def create_curriculum(
    strategy: str = "linear",
    total_epochs: int = 100,
    num_samples: int = 1000,
    **kwargs,
) -> Tuple[DifficultyScorer, CurriculumScheduler]:
    """
    Create curriculum learning components.

    Args:
        strategy: Curriculum strategy
        total_epochs: Total training epochs
        num_samples: Number of samples
        **kwargs: Additional config parameters

    Returns:
        Tuple of (difficulty_scorer, scheduler)
    """
    config = CurriculumConfig(strategy=strategy, **kwargs)

    # Select difficulty scorer
    if strategy in ["loss", "linear", "exponential"]:
        scorer = LossBasedDifficulty()
    elif strategy == "confidence":
        scorer = ConfidenceBasedDifficulty()
    elif strategy == "gradient":
        scorer = GradientNormDifficulty()
    elif strategy == "molecular":
        scorer = MolecularComplexityDifficulty()
    else:
        scorer = LossBasedDifficulty()

    scheduler = CurriculumScheduler(config, total_epochs, num_samples)

    return scorer, scheduler
