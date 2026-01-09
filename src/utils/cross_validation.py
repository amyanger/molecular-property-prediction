"""Cross-validation utilities for molecular property prediction."""

import numpy as np
from typing import Optional, Generator
from sklearn.model_selection import KFold, StratifiedKFold
from dataclasses import dataclass


@dataclass
class CVFold:
    """Container for cross-validation fold data."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    n_train: int
    n_val: int


@dataclass
class CVResults:
    """Container for cross-validation results."""

    n_folds: int
    fold_aucs: list[float]
    fold_losses: list[float]
    mean_auc: float
    std_auc: float
    mean_loss: float
    std_loss: float
    best_fold: int
    best_auc: float


def create_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Generator[CVFold, None, None]:
    """
    Create cross-validation splits.

    Args:
        n_samples: Number of samples
        n_folds: Number of folds
        shuffle: Whether to shuffle data
        random_state: Random seed

    Yields:
        CVFold objects for each fold
    """
    kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    indices = np.arange(n_samples)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        yield CVFold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )


def create_stratified_cv_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Generator[CVFold, None, None]:
    """
    Create stratified cross-validation splits for binary classification.

    Args:
        labels: Binary labels for stratification
        n_folds: Number of folds
        shuffle: Whether to shuffle data
        random_state: Random seed

    Yields:
        CVFold objects for each fold
    """
    skfold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(np.zeros(len(labels)), labels)):
        yield CVFold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )


def create_multitask_cv_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    stratify_task: int = 0,
    shuffle: bool = True,
    random_state: int = 42,
) -> Generator[CVFold, None, None]:
    """
    Create cross-validation splits for multi-task learning.

    Stratifies based on one task while handling missing labels.

    Args:
        labels: Multi-task labels (n_samples, n_tasks)
        n_folds: Number of folds
        stratify_task: Task index to use for stratification
        shuffle: Whether to shuffle data
        random_state: Random seed

    Yields:
        CVFold objects for each fold
    """
    # Get valid labels for stratification task
    task_labels = labels[:, stratify_task]
    valid_mask = task_labels != -1

    if valid_mask.sum() < n_folds * 2:
        # Fall back to regular CV if not enough valid samples
        yield from create_cv_splits(len(labels), n_folds, shuffle, random_state)
        return

    # Use valid samples for stratification
    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(~valid_mask)[0]

    skfold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    valid_labels = task_labels[valid_mask]

    for fold_idx, (train_idx_local, val_idx_local) in enumerate(
        skfold.split(np.zeros(len(valid_indices)), valid_labels)
    ):
        # Map back to global indices
        train_idx = valid_indices[train_idx_local]
        val_idx = valid_indices[val_idx_local]

        # Distribute invalid samples
        np.random.seed(random_state + fold_idx)
        shuffled_invalid = invalid_indices.copy()
        np.random.shuffle(shuffled_invalid)

        split_point = int(len(shuffled_invalid) * 0.8)
        train_idx = np.concatenate([train_idx, shuffled_invalid[:split_point]])
        val_idx = np.concatenate([val_idx, shuffled_invalid[split_point:]])

        # Shuffle combined indices
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        yield CVFold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=val_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )


def aggregate_cv_results(
    fold_aucs: list[float],
    fold_losses: Optional[list[float]] = None,
) -> CVResults:
    """
    Aggregate results from cross-validation folds.

    Args:
        fold_aucs: AUC scores for each fold
        fold_losses: Optional loss values for each fold

    Returns:
        CVResults with aggregated statistics
    """
    if fold_losses is None:
        fold_losses = [0.0] * len(fold_aucs)

    return CVResults(
        n_folds=len(fold_aucs),
        fold_aucs=fold_aucs,
        fold_losses=fold_losses,
        mean_auc=float(np.mean(fold_aucs)),
        std_auc=float(np.std(fold_aucs)),
        mean_loss=float(np.mean(fold_losses)),
        std_loss=float(np.std(fold_losses)),
        best_fold=int(np.argmax(fold_aucs)),
        best_auc=float(max(fold_aucs)),
    )


class CrossValidator:
    """
    Cross-validation manager for training and evaluation.

    Usage:
        cv = CrossValidator(n_folds=5)
        for fold in cv.get_folds(X, y):
            # Train model on fold.train_indices
            # Evaluate on fold.val_indices
            cv.record_fold_result(auc=0.85, loss=0.3)
        results = cv.get_results()
    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratified: bool = False,
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified

        self.fold_aucs = []
        self.fold_losses = []
        self.fold_metrics = []

    def get_folds(
        self,
        n_samples: int,
        labels: Optional[np.ndarray] = None,
    ) -> Generator[CVFold, None, None]:
        """
        Generate cross-validation folds.

        Args:
            n_samples: Number of samples (if labels not provided)
            labels: Optional labels for stratification

        Yields:
            CVFold objects
        """
        if self.stratified and labels is not None:
            if labels.ndim == 1:
                yield from create_stratified_cv_splits(
                    labels, self.n_folds, self.shuffle, self.random_state
                )
            else:
                yield from create_multitask_cv_splits(
                    labels, self.n_folds, 0, self.shuffle, self.random_state
                )
        else:
            yield from create_cv_splits(
                n_samples, self.n_folds, self.shuffle, self.random_state
            )

    def record_fold_result(
        self,
        auc: float,
        loss: Optional[float] = None,
        metrics: Optional[dict] = None,
    ):
        """Record results for a completed fold."""
        self.fold_aucs.append(auc)
        self.fold_losses.append(loss or 0.0)
        if metrics:
            self.fold_metrics.append(metrics)

    def get_results(self) -> CVResults:
        """Get aggregated cross-validation results."""
        return aggregate_cv_results(self.fold_aucs, self.fold_losses)

    def reset(self):
        """Reset for new cross-validation run."""
        self.fold_aucs = []
        self.fold_losses = []
        self.fold_metrics = []
